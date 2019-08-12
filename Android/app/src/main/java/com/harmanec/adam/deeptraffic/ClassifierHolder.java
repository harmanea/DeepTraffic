package com.harmanec.adam.deeptraffic;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.support.annotation.NonNull;

import java.io.IOException;
import java.util.List;

public class ClassifierHolder {
    private static final int INPUT_SIZE = 32;
    private static final long[] INPUT_DIMS = {1, 32, 32, 3}; // [batch_size, width, height, channels]
    private static final String INPUT_NAME = "conv_1.1_input";
    private static final String OUTPUT_NAME = "dense_softmax/Softmax";

    private static final String MODEL_FILE = "file:///android_asset/model.pb";
    private static final String LABEL_FILE = "file:///android_asset/labels.txt";

    private static Classifier classifier;

    public static void init(@NonNull AssetManager assetManager) throws IOException {
        classifier = TensorFlowImageClassifier.create(assetManager, MODEL_FILE, LABEL_FILE,
                INPUT_NAME, OUTPUT_NAME, INPUT_DIMS);
    }

    public static List<Classifier.Recognition> getPredictions(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        scaledBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        float[] image = new float[pixels.length * 3];
        for (int j = 0; j < pixels.length; j++) {
            // unpack rgb values
            int red = (pixels[j] >> 16) & 0xFF;
            int green = (pixels[j] >> 8) & 0xFF;
            int blue = pixels[j] & 0xFF;

            // normalize
            image[j * 3] = (float) (red / 255.0);
            image[j * 3 + 1] = (float) (green / 255.0);
            image[j * 3 + 2] = (float) (blue / 255.0);
        }

        return classifier.recognizeImage(image);
    }
}
