package com.facedetection.app;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button train = (Button)findViewById(R.id.btn_train);
        train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent swap = new Intent(MainActivity.this, TrainActivity.class);
                startActivity(swap);
            }
        });
        Button recognize = (Button)findViewById(R.id.btn_recognize);
        recognize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent swap = new Intent(MainActivity.this, RecognizeActivity.class);
                startActivity(swap);
            }
        });
    }
}
