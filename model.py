#  -*- coding: utf-8 -*

import tensorflow as tf
import datasets


model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=41, output_dim=128, input_length=75),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1 ,padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
        tf.keras.layers.GRU(units=128),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

checkpoint_path = os.path.join(r'./checkpoint')
tensorboard_path = os.path.join(r'./tensorboard')
batch_size = 32
epoch = 20

checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                        monitor='val_loss', 
                        verbose=0, 
                        save_best_only=True, 
                        save_weights_only=False, 
                        mode='auto', 
                        period=1)

tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=tensorboard_path,
                    histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None,
                    embeddings_data=None,
                    update_freq='epoch')

callbacks = [checkpoint, tensorboard]


train_set = datasets.csv_dataset_reader('train_ord_encode', batch_size, epoch)
val_set = datasets.csv_dataset_reader('val_ord_encode', batch_size, epoch)
test_set = datasets.csv_dataset_reader('test_ord_encode', batch_size, epoch)

model.fit(train_set, epochs=20, callbacks=callbacks, 
            validation_data=val_set, steps_per_epoch = 1139524 // 32, 
            validation_steps = 284881 // 32) 