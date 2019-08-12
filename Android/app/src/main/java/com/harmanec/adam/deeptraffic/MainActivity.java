package com.harmanec.adam.deeptraffic;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.soundcloud.android.crop.Crop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executor;

import static com.harmanec.adam.deeptraffic.DeepTraffic.REQUEST_CROP_IMAGE;
import static com.harmanec.adam.deeptraffic.DeepTraffic.REQUEST_IMAGE_CAPTURE;
import static com.harmanec.adam.deeptraffic.DeepTraffic.REQUEST_SELECT_FROM_GALLERY;
import static java.util.concurrent.Executors.newSingleThreadExecutor;

/**
 * Starting activity with most of the apps functionality
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private ImageView imageView;
    private ProgressBar[] progressBars;
    private TextView[] textViews;

    private Executor executor = newSingleThreadExecutor();

    private Uri captureImageDestination;
    private Uri croppedImageDestination;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.result);

        progressBars = new ProgressBar[] {
                findViewById(R.id.progressBar1),
                findViewById(R.id.progressBar2),
                findViewById(R.id.progressBar3)
        };
        textViews = new TextView[] {
                findViewById(R.id.textView1),
                findViewById(R.id.textView2),
                findViewById(R.id.textView3)
        };

        // Set imageView height to be equivalent to it's width
        Point size = new Point();
        getWindowManager().getDefaultDisplay().getSize(size);
        int swidth = size.x;

        ViewGroup.LayoutParams params = imageView.getLayoutParams();
        params.width = ViewGroup.LayoutParams.MATCH_PARENT;
        params.height = swidth;
        imageView.setLayoutParams(params);

        // Set up Take a picture Button
        findViewById(R.id.pictureButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                requestImageCapture();
            }
        });

        //Set up Select from gallery Button
        findViewById(R.id.galleryButton).setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                requestSelectFromGallery();
            }
        });

        // Add toolbar
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        // Init models and files
        initTensorflow();
        createTempFiles();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == R.id.info) {// Launch the InfoActivity when info button in the toolbar is pressed
            startActivity(new Intent(this, InfoActivity.class));
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestedCode, int resultCode, Intent intent) {
        if (resultCode == RESULT_OK) {
            switch (requestedCode) {
                case REQUEST_IMAGE_CAPTURE:
                    requestCrop(captureImageDestination);
                    break;
                case REQUEST_SELECT_FROM_GALLERY:
                    Uri imageUri = intent.getData();
                    requestCrop(imageUri);
                    break;
                case REQUEST_CROP_IMAGE:
                    try {
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), croppedImageDestination);
                        useBitmap(bitmap);
                    } catch (IOException e) {
                        Log.e(TAG, "Error obtaining bitmap from cropped image.", e);
                    }
                default:
                    Log.d(TAG, "Unknown request code " + requestedCode);

            }

        } else if (resultCode == RESULT_CANCELED) {
            Log.d(TAG, "Action cancelled");
        }
    }

    private void initTensorflow() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    ClassifierHolder.init(getAssets());
                    Log.d(TAG, "Tensorflow loaded successfully");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void createTempFiles() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    captureImageDestination = createImageFile("capturedImage");
                    croppedImageDestination = createImageFile("croppedImage");
                    Log.d(TAG, "Temp files created successfully");
                } catch (final IOException e) {
                    throw new RuntimeException("Error creating temp files!", e);
                }
            }
        });
    }

    private Uri createImageFile(String prefix) throws IOException {
        @SuppressLint("SimpleDateFormat") String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = prefix + "_" + timeStamp;
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,
                ".jpg",
                storageDir
        );

        return FileProvider.getUriForFile(MainActivity.this,
                "com.harmanec.adam.deeptraffic.fileprovider",
                image);
    }

    private void requestImageCapture() {
        Intent captureImageIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        if (captureImageIntent.resolveActivity(getPackageManager()) != null) {
            captureImageIntent.putExtra(MediaStore.EXTRA_OUTPUT, captureImageDestination);
            startActivityForResult(captureImageIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void requestSelectFromGallery() {
        Intent selectFromGalleryIntent = new Intent(Intent.ACTION_GET_CONTENT);
        selectFromGalleryIntent.setType("image/*");
        if (selectFromGalleryIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(selectFromGalleryIntent, REQUEST_SELECT_FROM_GALLERY);
        }
    }

    private void requestCrop(Uri srcUri) {
        Crop.of(srcUri, croppedImageDestination)
                .start(this, REQUEST_CROP_IMAGE);
    }

    private void useBitmap(Bitmap bitmap) {
        imageView.setImageBitmap(squareBitmap(bitmap));

        List<Classifier.Recognition> predictions = ClassifierHolder.getPredictions(bitmap);

        for (int i = 0; i < 3; i++) {

            final int confidence = Math.round(predictions.get(i).getConfidence() * 100);
            progressBars[i].setProgress(confidence);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                progressBars[i].setTooltipText(confidence + "%");
            }
            textViews[i].setText(predictions.get(i).getTitle().trim());
        }
    }

    /**
     * Crop Bitmap to square
     * @param src Bitmap to be cropped
     * @return cropped Bitmap
     */
    private Bitmap squareBitmap(Bitmap src) {
        int size = Math.min(src.getWidth(), src.getHeight());
        return Bitmap.createScaledBitmap(src, size, size, true);
    }

}
