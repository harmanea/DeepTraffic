package com.harmanec.adam.deeptraffic;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Starting activity with most of the apps functionality
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private static final int REQUEST_CODE = 1;
    private TextView textView;
    private ImageView imageView;

    private static final int MODEL_COUNT = 3;
    private static int model = 0;

    private static final int[] INPUT_SIZES = new int[]{32, 32, 96};
    private static final long[][] INPUT_DIMS = {{1, 32, 32, 1}, {1, 32, 32, 1}, {1, 96, 96, 3}}; // [batch_size, width, height, channels]
    private static final String[] INPUT_NAMES = new String[] {"conv2d_input", "flatten_input", "module_apply_default/hub_input/Mul"};
    private static final String[] OUTPUT_NAMES = new String[] {"dense_1/Softmax", "dense_2/Softmax", "final_result"};

    private static final String[] MODEL_FILES = new String[] {"file:///android_asset/traffic.pb", "file:///android_asset/traffic_simple.pb", "file:///android_asset/retrained_MobileNetV2.pb"};
    private static final String[] LABEL_FILES = new String[] {"file:///android_asset/image_labels.txt", "file:///android_asset/image_labels.txt", "file:///android_asset/image_labels.txt"};

    private String[] predictions = new String[MODEL_COUNT];

    private Classifier[] classifiers = new Classifier[MODEL_COUNT];
    private Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.textView);
        imageView = findViewById(R.id.result);

        // Set textView height to be equivalent to it's width
        Point size = new Point();
        getWindowManager().getDefaultDisplay().getSize(size);
        int swidth = size.x;

        ViewGroup.LayoutParams params = imageView.getLayoutParams();
        params.width = ViewGroup.LayoutParams.MATCH_PARENT;
        params.height = swidth;
        imageView.setLayoutParams(params);

        // Set up Button
        Button button = findViewById(R.id.pictureButton);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, REQUEST_CODE);
            }
        });

        // Set up Spinner
        Spinner spinner = findViewById(R.id.modelSpinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.models, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                model = position;
                if (predictions[model] != null) {
                    textView.setText(predictions[model]);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

        // Add toolbar
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        initTensorflow();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.info:
                // Launch the InfoActivity when info button in the toolbar is pressed
                startActivity(new Intent(this, InfoActivity.class));
                return true;

            default:
                return super.onOptionsItemSelected(item);
        }
    }

    /**
     * Load and set up tensorflow classifiers
     */
    private void initTensorflow() {
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        for (int i = 0; i < MODEL_COUNT; i++) {
                            classifiers[i] = TensorFlowImageClassifier.create(
                                    getAssets(),
                                    MODEL_FILES[i],
                                    LABEL_FILES[i],
                                    INPUT_NAMES[i],
                                    OUTPUT_NAMES[i],
                                    INPUT_DIMS[i]);
                            Log.d(TAG, "Model from file " + MODEL_FILES[i] + " loaded successfully");
                        }
                        Log.d(TAG, "Tensorflow loaded successfully");
                    } catch (final Exception e){
                        throw new RuntimeException("Error initializing TensorFlow!", e);
                    }
                }
            });
    }

    @Override
    protected void onActivityResult(int requestedCode, int resultCode, Intent data) {
        if (requestedCode == REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            // Get the taken photo, crop it to square and display it
            try {
                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                bitmap = squareBitmap(bitmap);
                imageView.setImageBitmap(bitmap);

                // Get predictions on the photo and display the selected one
                getPredictions(bitmap);
                textView.setText(predictions[model]);
            } catch (NullPointerException e) {
                Log.d(TAG, "Extras.get('data') returned null");
            }

        } else if (resultCode == Activity.RESULT_CANCELED) {
            Log.d(TAG, "Camera Intent cancelled");
        }

    }

    /**
     * Get predictions on the provided Bitmap for all available classifiers and save them to array predictions
     * @param bmp Bitmap to analyze
     */
    private void getPredictions(Bitmap bmp) {
        for (int i = 0; i < MODEL_COUNT; i++) {
            // scale bitmap and extract pixel values
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bmp, INPUT_SIZES[i], INPUT_SIZES[i], true);
            int pixels[] = new int[INPUT_SIZES[i] * INPUT_SIZES[i]];
            scaledBitmap.getPixels(pixels, 0, INPUT_SIZES[i], 0, 0, INPUT_SIZES[i], INPUT_SIZES[i]);

            float image[];
            if (INPUT_DIMS[i][3] == 1) { // if model takes only one grayscale channel
                // convert to grayscale and normalize to float values between 0 and 1
                image = new float[pixels.length];
                for (int j = 0; j < pixels.length; j++) {
                    // unpack rgb values
                    int red = (pixels[j] >> 16) & 0xFF;
                    int green = (pixels[j] >> 8) & 0xFF;
                    int blue = pixels[j] & 0xFF;

                    // grayscale YCbCr representation
                    image[j] = (float) (0.299 * red + 0.587 * green + 0.114 * blue);

                    // normalize
                    image[j] = (float) (image[j] / 255.0);
                }
            } else {
                // normalize to float values between 0 and 1
                image = new float[pixels.length * 3];
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
            }

            // feed preprocessed data to classifier
            final List<Classifier.Recognition> results = classifiers[i].recognizeImage(image);
            if (results.size() > 0) {
                // save top result
                predictions[i] = results.get(0).getTitle() + " (" + Math.round(results.get(0).getConfidence() * 100) + "%)";
            } else {
                predictions[i] = "Not recognized";
            }
        }
    }

    /**
     * Crop Bitmap to square
     * @param src Bitmap to be cropped
     * @return cropped Bitmap
     */
    private Bitmap squareBitmap(Bitmap src) {
        boolean portrait = src.getHeight() > src.getWidth();
        int size = portrait ? src.getWidth() : src.getHeight();
        int x_offset = portrait ? 0 : ((src.getWidth() - src.getHeight()) / 2);
        int y_offset = portrait ? ((src.getHeight() - src.getWidth()) / 2) : 0;
        return Bitmap.createBitmap(src, x_offset, y_offset, size, size);
    }
}
