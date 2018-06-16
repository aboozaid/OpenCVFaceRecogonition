package com.facedetection.app;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;

/**
 * Created by Assem Abozaid on 6/3/2018.
 */

public class FileUtils {
    private static String TAG = FileUtils.class.getSimpleName();
    private static boolean loadFile(Context context, String cascadeName) {
        InputStream inp = null;
        OutputStream out = null;
        boolean completed = false;
        try {
            inp = context.getResources().getAssets().open(cascadeName);
            File outFile = new File(context.getCacheDir(), cascadeName);
            out = new FileOutputStream(outFile);

            byte[] buffer = new byte[4096];
            int bytesread;
            while((bytesread = inp.read(buffer)) != -1) {
                out.write(buffer, 0, bytesread);
            }

            completed = true;
            inp.close();
            out.flush();
            out.close();
        } catch (IOException e) {
            Log.i(TAG, "Unable to load cascade file" + e);
        }
        return completed;
    }
    public static CascadeClassifier loadXMLS(Context context, String cascadeName) {

        CascadeClassifier classifier = null;

        if(loadFile(context, cascadeName)) {
            File cascade = new File(context.getCacheDir(), cascadeName);
            classifier = new CascadeClassifier(cascade.getAbsolutePath());
            Log.i(TAG, "Cascade File Loaded Successfully");
            cascade.delete();
        } else {
            Log.i(TAG, "Path Direction May Be Wrong");
        }
        return classifier;
    }
    public static String loadTrained() {
        File file = new File(Environment.getExternalStorageDirectory(), "TrainedData/lbph_trained_data.xml");

        return file.toString();
    }
}
