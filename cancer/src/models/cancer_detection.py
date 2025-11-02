"""
Modelos de Deep Learning para detección temprana de cáncer.
Implementa arquitecturas CNN, Vision Transformers y modelos híbridos.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import warnings

import numpy as np

# Importaciones opcionales para deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import (
        ResNet50, EfficientNetB0, VGG16, InceptionV3, DenseNet121
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow no está instalado. Instale con: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch no está instalado. Instale con: pip install torch torchvision")

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn no está instalado. Instale con: pip install scikit-learn")

TF_MISSING_MSG = "TensorFlow no está disponible"


class CancerDetectionModel:
    """Modelo base para detección de cáncer usando deep learning."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el modelo de detección de cáncer.
        
        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "config.json")
        
        with open(str(config_path), 'r') as f:
            self.config = json.load(f)
        
        # Configurar logging
        logging.basicConfig(level=getattr(logging, self.config['logging']['level']))
        self.logger = logging.getLogger(__name__)
        
        # Parámetros del modelo
        self.model_config = self.config['model']['early_detection']
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']
        self.learning_rate = self.model_config['learning_rate']
        self.epochs = self.model_config['epochs']
        self.patience = self.model_config['patience']
        
        self.model = None
        self.history = None
        
    def create_cnn_model(self, architecture: str = "ResNet50") -> Any:
        """
        Crea un modelo CNN para detección de cáncer.
        
        Args:
            architecture: Arquitectura del modelo ('ResNet50', 'EfficientNetB0', etc.)
            
        Returns:
            Modelo compilado
        """
        if not TF_AVAILABLE:
            self.logger.error(TF_MISSING_MSG)
            return None
        
        try:
            # Seleccionar arquitectura base
            if architecture == "ResNet50":
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            elif architecture == "EfficientNetB0":
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            elif architecture == "VGG16":
                base_model = VGG16(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            elif architecture == "DenseNet121":
                base_model = DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            else:
                self.logger.error(f"Arquitectura no soportada: {architecture}")
                return None
            
            # Congelar capas base inicialmente
            base_model.trainable = False
            
            # Crear modelo completo
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
            ])
            
            # Compilar modelo
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            self.logger.info(f"Modelo {architecture} creado exitosamente")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creando modelo {architecture}: {e}")
            return None
    
    def create_vision_transformer(self, patch_size: int = 16, num_heads: int = 8, 
                                 num_layers: int = 12) -> Any:
        """
        Crea un Vision Transformer para detección de cáncer.
        
        Args:
            patch_size: Tamaño de patches
            num_heads: Número de cabezas de atención
            num_layers: Número de capas transformer
            
        Returns:
            Modelo Vision Transformer
        """
        if not TF_AVAILABLE:
            self.logger.error(TF_MISSING_MSG)
            return None
        
        try:
            # Parámetros del modelo
            image_size = self.input_shape[0]  # Asumiendo imagen cuadrada
            num_patches = (image_size // patch_size) ** 2
            projection_dim = 768
            transformer_units = [projection_dim * 2, projection_dim]
            
            # Inputs
            inputs = layers.Input(shape=self.input_shape)
            
            # Crear patches
            patches = self._extract_patches(inputs, patch_size)
            
            # Proyección lineal de patches
            encoded_patches = layers.Dense(units=projection_dim)(patches)
            
            # Agregar embeddings posicionales
            positions = tf.range(start=0, limit=num_patches, delta=1)
            position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )(positions)
            encoded_patches += position_embedding
            
            # Transformer blocks
            for _ in range(num_layers):
                # Multi-head attention 
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                attention_output = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=projection_dim
                )(x1, x1)
                x2 = layers.Add()([attention_output, encoded_patches])
                
                # MLP
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                x3 = self._mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
                encoded_patches = layers.Add()([x3, x2])
            
            # Clasificación
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = layers.GlobalAveragePooling1D()(representation)
            representation = layers.Dropout(0.5)(representation)
            
            # Capas finales
            features = layers.Dense(512, activation="gelu")(representation)
            features = layers.Dropout(0.3)(features)
            logits = layers.Dense(self.num_classes)(features)
            
            if self.num_classes == 2:
                outputs = layers.Activation("sigmoid")(logits)
            else:
                outputs = layers.Activation("softmax")(logits)
            
            # Crear modelo
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Compilar
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info("Vision Transformer creado exitosamente")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creando Vision Transformer: {e}")
            return None
    
    def create_hybrid_model(self) -> Any:
        """
        Crea un modelo híbrido CNN + Vision Transformer.
        
        Returns:
            Modelo híbrido
        """
        if not TF_AVAILABLE:
            self.logger.error(TF_MISSING_MSG)
            return None
        
        try:
            inputs = layers.Input(shape=self.input_shape)
            
            # Rama CNN
            cnn_base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            cnn_base.trainable = False
            
            cnn_features = cnn_base(inputs)
            cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
            cnn_features = layers.Dense(256, activation='relu')(cnn_features)
            
            # Rama ViT (simplificada)
            vit_patches = self._extract_patches(inputs, patch_size=32)
            vit_encoded = layers.Dense(256)(vit_patches)
            vit_attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(vit_encoded, vit_encoded)
            vit_features = layers.GlobalAveragePooling1D()(vit_attention)
            vit_features = layers.Dense(256, activation='relu')(vit_features)
            
            # Fusión de características
            combined = layers.Concatenate()([cnn_features, vit_features])
            combined = layers.Dense(512, activation='relu')(combined)
            combined = layers.Dropout(0.5)(combined)
            combined = layers.Dense(256, activation='relu')(combined)
            combined = layers.Dropout(0.3)(combined)
            
            # Clasificación
            if self.num_classes == 2:
                outputs = layers.Dense(1, activation='sigmoid')(combined)
            else:
                outputs = layers.Dense(self.num_classes, activation='softmax')(combined)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Compilar
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.logger.info("Modelo híbrido creado exitosamente")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creando modelo híbrido: {e}")
            return None
    
    def _extract_patches(self, images, patch_size):
        """Extrae patches de imágenes para Vision Transformer."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def _mlp(self, x, hidden_units, dropout_rate):
        """Bloque MLP para Vision Transformer."""
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def train_model(self, train_data, validation_data, model_type: str = "ResNet50") -> Dict[str, Any]:
        """
        Entrena el modelo de detección de cáncer.
        
        Args:
            train_data: Datos de entrenamiento (X_train, y_train)
            validation_data: Datos de validación (X_val, y_val)
            model_type: Tipo de modelo a entrenar
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        if not TF_AVAILABLE:
            self.logger.error(TF_MISSING_MSG)
            return {}
        
        try:
            X_train, y_train = train_data
            x_val, y_val = validation_data
            
            # Crear modelo según el tipo
            if model_type == "ViT":
                self.model = self.create_vision_transformer()
            elif model_type == "Hybrid":
                self.model = self.create_hybrid_model()
            else:
                self.model = self.create_cnn_model(model_type)
            
            if self.model is None:
                return {"error": "No se pudo crear el modelo"}
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    filepath=f'best_model_{model_type}.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Entrenar modelo
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=self.epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluar modelo
            train_metrics = self.evaluate_model(X_train, y_train)
            val_metrics = self.evaluate_model(x_val, y_val)
            
            results = {
                'model_type': model_type,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'history': self.history.history,
                'best_epoch': len(self.history.history['loss']) - self.patience
            }
            
            self.logger.info(f"Entrenamiento completado para {model_type}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error durante entrenamiento: {e}")
            return {"error": str(e)}
    
    def evaluate_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evalúa el modelo entrenado.
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            Métricas de evaluación
        """
        if self.model is None:
            self.logger.error("No hay modelo entrenado")
            return {}
        
        try:
            # Predicciones
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int) if self.num_classes == 2 else np.argmax(y_pred_proba, axis=1)
            
            # Métricas
            if SKLEARN_AVAILABLE:
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
                
                # AUC solo para clasificación binaria
                if self.num_classes == 2:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                
                return metrics
            else:
                # Métricas básicas sin sklearn
                accuracy = np.mean(y_pred == y_test)
                return {'accuracy': accuracy}
                
        except Exception as e:
            self.logger.error(f"Error en evaluación: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Predicciones con probabilidades
        """
        if self.model is None:
            self.logger.error("No hay modelo entrenado")
            return {}
        
        try:
            predictions_proba = self.model.predict(X)
            
            if self.num_classes == 2:
                predictions = (predictions_proba > 0.5).astype(int)
                confidence = np.max([predictions_proba, 1 - predictions_proba], axis=0)
            else:
                predictions = np.argmax(predictions_proba, axis=1)
                confidence = np.max(predictions_proba, axis=1)
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': predictions_proba.tolist(),
                'confidence': confidence.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
            
        Returns:
            True si se guardó exitosamente
        """
        if self.model is None:
            self.logger.error("No hay modelo para guardar")
            return False
        
        try:
            self.model.save(filepath)
            self.logger.info(f"Modelo guardado en: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta del modelo a cargar
            
        Returns:
            True si se cargó exitosamente
        """
        if not TF_AVAILABLE:
            self.logger.error(TF_MISSING_MSG)
            return False
        
        try:
            self.model = keras.models.load_model(filepath)
            self.logger.info(f"Modelo cargado desde: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            return False
    
    def fine_tune(self, train_data, validation_data, unfreeze_layers: int = 20) -> Dict[str, Any]:
        """
        Realiza fine-tuning del modelo.
        
        Args:
            train_data: Datos de entrenamiento
            validation_data: Datos de validación
            unfreeze_layers: Número de capas a descongelar
            
        Returns:
            Resultados del fine-tuning
        """
        if self.model is None:
            self.logger.error("No hay modelo para fine-tuning")
            return {}
        
        try:
            # Descongelar capas superiores del modelo base
            if hasattr(self.model.layers[0], 'trainable'):
                self.model.layers[0].trainable = True
                
                # Congelar todas las capas excepto las últimas
                for layer in self.model.layers[0].layers[:-unfreeze_layers]:
                    layer.trainable = False
            
            # Recompilar con learning rate más bajo
            optimizer = optimizers.Adam(learning_rate=self.learning_rate / 10)
            self.model.compile(
                optimizer=optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics
            )
            
            # Fine-tuning con menos épocas
            X_train, y_train = train_data
            x_val, y_val = validation_data
            
            fine_tune_history = self.model.fit(
                X_train, y_train,
                batch_size=16,  # Batch size más pequeño para fine-tuning
                epochs=self.epochs // 2,
                validation_data=(x_val, y_val),
                verbose=1
            )
            
            # Evaluar después del fine-tuning
            val_metrics = self.evaluate_model(x_val, y_val)
            
            results = {
                'fine_tune_history': fine_tune_history.history,
                'validation_metrics': val_metrics
            }
            
            self.logger.info("Fine-tuning completado")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en fine-tuning: {e}")
            return {}


def load_cancer_detection_model():
    """Función helper para cargar el modelo con configuración por defecto."""
    return CancerDetectionModel()


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar modelo
    model = CancerDetectionModel()
    
    # Ejemplo de creación de modelo (requiere datos reales)
    # cnn_model = model.create_cnn_model("ResNet50")
    # vit_model = model.create_vision_transformer()
    # hybrid_model = model.create_hybrid_model()
    
    print("Modelo de detección de cáncer inicializado correctamente")
    print(f"TensorFlow disponible: {TF_AVAILABLE}")
    print(f"PyTorch disponible: {TORCH_AVAILABLE}")
    print(f"scikit-learn disponible: {SKLEARN_AVAILABLE}")