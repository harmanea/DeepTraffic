package com.harmanec.adam.deeptraffic;

import android.os.Bundle;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.Html;
import android.text.Spanned;
import android.text.method.ScrollingMovementMethod;
import android.widget.TextView;

/**
 * Simple Activity to show the info text
 */
public class InfoActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info);

        // Add toolbar
        Toolbar toolbar = findViewById(R.id.child_toolbar);
        setSupportActionBar(toolbar);

        ActionBar ab = getSupportActionBar();
        // Enable up button
        assert ab != null;
        ab.setDisplayHomeAsUpEnabled(true);
        ab.setTitle("Help");

        // Set text
        TextView textView = findViewById(R.id.infoText);
        Spanned s = Html.fromHtml(getString(R.string.info_text));
        textView.setText(s);
        textView.setMovementMethod(new ScrollingMovementMethod());
    }
}
