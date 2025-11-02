"""
Modelo multimodal de fusión (imagen + radiómico opcional) para detección de cáncer.
Incluye entrenamiento con K-Fold CV, data augmentation y guardado de artefactos.
Configuración centralizada en config.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks, models
    from tensorflow.keras.applications import EfficientNetB0, ResNet50
    TF_AVAILABLE = True
except Exception:  # noqa: BLE001
    TF_AVAILABLE = False


def _load_fusion_config() -> Dict[str, Any]:
    """Carga configuración de fusion desde config.json."""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                return cfg.get('model', {}).get('fusion', {})
        except Exception:  # noqa: BLE001
            pass
    return {}


@dataclass
class FusionConfig:
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    radiomics_dim: int = 0  # Si 0, no se usa rama radiómica
    base_model: str = "EfficientNetB0"  # o "ResNet50"
    learning_rate: float = 1e-3
    epochs: int = 15
    batch_size: int = 32
    patience: int = 7
    k_folds: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    results_dir: Path = Path(__file__).parent.parent.parent / "results"
    
    @classmethod
    def from_config_file(cls) -> "FusionConfig":
        """Crea FusionConfig desde config.json."""
        cfg_dict = _load_fusion_config()
        if not cfg_dict:
            return cls()
        
        return cls(
            input_shape=tuple(cfg_dict.get('input_shape', [224, 224, 3])),
            num_classes=cfg_dict.get('num_classes', 2),
            radiomics_dim=cfg_dict.get('radiomics_dim', 0),
            base_model=cfg_dict.get('base_model', 'EfficientNetB0'),
            learning_rate=cfg_dict.get('learning_rate', 1e-3),
            epochs=cfg_dict.get('epochs', 15),
            batch_size=cfg_dict.get('batch_size', 32),
            patience=cfg_dict.get('patience', 7),
            k_folds=cfg_dict.get('k_folds', 5),
            reduce_lr_patience=cfg_dict.get('reduce_lr_patience', 3),
            reduce_lr_factor=cfg_dict.get('reduce_lr_factor', 0.5),
            min_lr=cfg_dict.get('min_lr', 1e-6)
        )


class FusionCancerModel:
    def __init__(self, cfg: Optional[FusionConfig] = None) -> None:
        if cfg is None:
            cfg = FusionConfig.from_config_file()
        self.cfg = cfg
        self.model: Optional[keras.Model] = None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.cfg.results_dir / "models" / f"fusion_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _get_base(self) -> keras.Model:
        if self.cfg.base_model == "ResNet50":
            base = ResNet50(weights="imagenet", include_top=False, input_shape=self.cfg.input_shape)
        else:
            base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=self.cfg.input_shape)
        base.trainable = False
        return base

    def build_model(self) -> keras.Model:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow no está disponible")

        # Rama de imagen
        img_in = layers.Input(shape=self.cfg.input_shape, name="image")
        base = self._get_base()
        x = base(img_in)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

        # Rama radiómica opcional
        if self.cfg.radiomics_dim and self.cfg.radiomics_dim > 0:
            rad_in = layers.Input(shape=(self.cfg.radiomics_dim,), name="radiomics")
            r = layers.BatchNormalization()(rad_in)
            r = layers.Dense(128, activation="relu")(r)
            r = layers.Dropout(0.2)(r)
            fused = layers.Concatenate()([x, r])
            inputs = [img_in, rad_in]
        else:
            fused = x
            inputs = img_in

        # Cabeza de clasificación
        z = layers.Dense(256, activation="relu")(fused)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.4)(z)
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dropout(0.2)(z)

        if self.cfg.num_classes == 2:
            out = layers.Dense(1, activation="sigmoid", name="output")(z)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            out = layers.Dense(self.cfg.num_classes, activation="softmax", name="output")(z)
            loss = "categorical_crossentropy"
            metrics = ["accuracy"]

        model = keras.Model(inputs=inputs, outputs=out, name="fusion_cancer_model")
        opt = optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.model = model
        return model

    def _eval_predictions(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        if self.cfg.num_classes == 2:
            y_pred = (y_pred_proba.ravel() > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "classification_report": report,
            "confusion_matrix": cm,
        }

    # --------- tf.data pipelines ---------
    def _decode_image(self, path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, self.cfg.input_shape[:2])
        return img

    def _augment(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img

    def _make_ds(
        self,
        image_paths: Sequence[str],
        labels: np.ndarray,
        radiomics: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> tf.data.Dataset:
        img_ds = tf.data.Dataset.from_tensor_slices(list(map(str, image_paths)))
        img_ds = img_ds.map(self._decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            img_ds = img_ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        y_ds = tf.data.Dataset.from_tensor_slices(labels)

        if radiomics is not None and self.cfg.radiomics_dim and self.cfg.radiomics_dim > 0:
            r_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(radiomics, dtype=tf.float32))
            x_ds = tf.data.Dataset.zip(((img_ds, r_ds), y_ds))
        else:
            x_ds = tf.data.Dataset.zip((img_ds, y_ds))

        ds = x_ds.shuffle(1024) if training else x_ds
        ds = ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _train_evaluate_fold(
        self,
        *,
        fold: int,
        paths: List[str],
        y: np.ndarray,
        xr: Optional[np.ndarray],
        idx_tr: np.ndarray,
        idx_va: np.ndarray,
        class_names: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
        """Entrena y evalúa un fold, devolviendo historia, reporte y ruta de checkpoint."""
        # Split
        x_tr = [paths[i] for i in idx_tr]
        x_va = [paths[i] for i in idx_va]
        y_tr = y[idx_tr]
        y_va = y[idx_va]
        xr_tr = xr[idx_tr] if xr is not None else None
        xr_va = xr[idx_va] if xr is not None else None

        # Modelo y datasets
        self.build_model()
        model = self.model
        if model is None:
            raise RuntimeError("Error interno: el modelo no fue construido")
        train_ds = self._make_ds(x_tr, y_tr, xr_tr, training=True)
        val_ds = self._make_ds(x_va, y_va, xr_va, training=False)

        # Callbacks y entrenamiento
        ckpt_path = self.run_dir / f"fold{fold}_best.h5"
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=self.cfg.patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=self.cfg.reduce_lr_factor, 
                patience=self.cfg.reduce_lr_patience, 
                min_lr=self.cfg.min_lr
            ),
            callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_accuracy", save_best_only=True),
        ]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=self.cfg.epochs, verbose=1, callbacks=cbs)

        # Evaluación
        y_pred_proba = model.predict(val_ds)
        eval_res = self._eval_predictions(y_va, y_pred_proba, class_names)
        return hist.history, eval_res, ckpt_path

    # --------- entrenamiento con K-Fold ---------
    def fit_kfold(
        self,
        df,
        image_col: str,
        label_col: str,
        radiomics_cols: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow no está disponible")
        from sklearn.model_selection import StratifiedKFold

        def _prepare_targets_and_radiomics():
            paths_local = df[image_col].astype(str).tolist()
            y_raw_local = df[label_col].values

            # Mapear etiquetas a enteros si son strings
            if df[label_col].dtype == object:
                uniq_local = sorted(df[label_col].unique())
                mapping = {v: i for i, v in enumerate(uniq_local)}
                y_local = df[label_col].map(mapping).astype(int).values
                cls_local = class_names or uniq_local
            else:
                y_local = y_raw_local.astype(int)
                cls_local = class_names or [str(i) for i in sorted(np.unique(y_local))]

            xr_local = None
            if radiomics_cols:
                xr_local = df[radiomics_cols].astype(float).values
                self.cfg.radiomics_dim = xr_local.shape[1]
            return paths_local, y_local, xr_local, list(cls_local)

        paths, y, xr, class_names_out = _prepare_targets_and_radiomics()

        skf = StratifiedKFold(n_splits=self.cfg.k_folds, shuffle=True, random_state=seed)
        fold_histories = []
        fold_reports = []
        best_val_acc = -1.0
        best_model_path = None

        for fold, (idx_tr, idx_va) in enumerate(skf.split(paths, y), start=1):
            hist_dict, eval_res, ckpt_path = self._train_evaluate_fold(
                fold=fold, paths=paths, y=y, xr=xr, idx_tr=idx_tr, idx_va=idx_va, class_names=class_names_out
            )
            fold_histories.append(hist_dict)
            fold_reports.append({"fold": fold, **eval_res})

            # Guardar mejor fold por val_accuracy
            if eval_res["accuracy"] > best_val_acc:
                best_val_acc = eval_res["accuracy"]
                best_model_path = ckpt_path

        # Guardar artefactos
        summary = {
            "k_folds": self.cfg.k_folds,
            "histories": fold_histories,
            "fold_reports": fold_reports,
            "best_val_accuracy": float(best_val_acc),
            "best_model_path": str(best_model_path) if best_model_path else None,
            "class_names": class_names_out,
        }
        with open(self.run_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def load_best(self, path: str) -> bool:
        if not TF_AVAILABLE:
            return False
        try:
            self.model = keras.models.load_model(path)
            return True
        except Exception:  # noqa: BLE001
            return False

    def predict(self, image: np.ndarray, radiomics_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "modelo no cargado"}
        x = image.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        x = np.expand_dims(x, axis=0)
        if radiomics_vec is not None and self.cfg.radiomics_dim and self.cfg.radiomics_dim > 0:
            rv = np.asarray(radiomics_vec, dtype=np.float32)[None, :]
            proba = self.model.predict([x, rv])
        else:
            proba = self.model.predict(x)
        if self.cfg.num_classes == 2:
            p = float(np.ravel(proba)[0])
            return {"prob_maligno": p, "label": "Maligno" if p >= 0.5 else "Benigno"}
        else:
            idx = int(np.argmax(proba))
            return {"class": idx, "proba": proba.tolist()}
