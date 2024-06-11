package com.example.kilam.androidfacerecognition;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.constraint.ConstraintLayout;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.SearchView;
import android.widget.TextView;
import android.widget.Toast;
import android.os.Handler;
import android.os.Looper;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import com.example.kilam.androidfacerecognition.Custom.CameraBridgeViewBase;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
//import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final long EXPIRATION_TIME = TimeUnit.SECONDS.toMillis(30);
    private HashMap<String, Long> sentLabels;
    private static final String TAG = "MainActivityOpenCV";
    private static final String MODEL_PATH = "mobile_face_net.tflite";

    String confidenceValue = BuildConfig.CONFIDENCE_VALUE;

    private static final int ID_CAMERA = CameraBridgeViewBase.CAMERA_ID_BACK;

    private Mat mRGBA;
    private CameraBridgeViewBase javaCameraView;
    private CascadeClassifier cascadeClassifier;
    private File mCascadeFile;
    private Bitmap mBitmap;

    private MappedByteBuffer tfliteModel;
    private Interpreter interpreter;
    private TensorImage tImage;
    private TensorBuffer tBuffer;

    private ArrayList<Person> persons;

    private static final int inputImageWidth = 112;
    private static final int inputImageHeight = 112;

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 128f;
    private PersonDAO personDAO;
    private ExecutorService executorService;
    private RabbitMQHelper rabbitMQHelper;
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        sentLabels = new HashMap<>();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_detection);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
        }
        executorService = Executors.newFixedThreadPool(1);

        connectWithRetry();

        personDAO = new PersonDAO(this);
        persons = personDAO.getAllPersons();

        initializeModel();

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV is Configured or Connected Successfully");
        } else {
            Log.d(TAG, "OpenCV is not working or Loaded");
        }

        javaCameraView = findViewById(R.id.opencv_camera);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        findViewById(R.id.btn_register).setOnClickListener(view -> {
            if (mBitmap != null) showDialog();
            else Toast.makeText(MainActivity.this, "No face detected", Toast.LENGTH_SHORT).show();
        });
        findViewById(R.id.btn_remove).setOnClickListener(view -> showDeleteDialog());

    }

    private void showDialog() {
        onPause();

        float[] embedding = getEmbedding(mBitmap);
        String recognizedPerson = recognize(embedding);

        if (!recognizedPerson.equals("unknown")) {
            Toast.makeText(MainActivity.this, "Face was recognize as: " + recognizedPerson, Toast.LENGTH_SHORT).show();
            onResume();
            return;
        }

        Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setCancelable(false);
        dialog.setContentView(R.layout.dialog_register);
        Window window = dialog.getWindow();
        if (window != null) {
            window.setLayout(ConstraintLayout.LayoutParams.MATCH_PARENT, ConstraintLayout.LayoutParams.WRAP_CONTENT);
        }

        ImageView iv_person = dialog.findViewById(R.id.iv_capture);
        iv_person.setImageBitmap(mBitmap);

        EditText et_name = dialog.findViewById(R.id.et_name);
        EditText et_nip = dialog.findViewById(R.id.et_nip);
        Button btnSave = dialog.findViewById(R.id.btn_save);
        Button btnCancel = dialog.findViewById(R.id.btn_cancel);

        btnSave.setOnClickListener(view -> {
            String name = et_name.getText().toString().trim().toLowerCase();
            String nip = et_nip.getText().toString().trim();
            if (name.isEmpty() || nip.isEmpty()) {
                Toast.makeText(MainActivity.this, "Name or NIP cannot be empty", Toast.LENGTH_SHORT).show();
            }
            else if (personDAO.isNipExists(nip) || personDAO.isNameExist(name)) {
                Toast.makeText(MainActivity.this, "Name or NIP already exists", Toast.LENGTH_SHORT).show();
            }
            else {
                personDAO.addPerson(new Person(name, nip, embedding));
                Toast.makeText(MainActivity.this, "Person added with name: " + name, Toast.LENGTH_SHORT).show();
                dialog.dismiss();
                persons = personDAO.getAllPersons();
                onResume();
            }
        });

        btnCancel.setOnClickListener(view -> {
            dialog.dismiss();
            onResume();
        });

        dialog.show();
    }
    private void showDeleteDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = this.getLayoutInflater();
        View dialogView = inflater.inflate(R.layout.dialog_delete_person, null);
        builder.setView(dialogView);

        // Setup title
        TextView title = dialogView.findViewById(R.id.title);
        title.setText("Delete Person");

        // Setup SearchView
        SearchView searchView = dialogView.findViewById(R.id.search_view);

        // Setup ListView
        ListView personListView = dialogView.findViewById(R.id.person_list);
        ArrayList<String> nameList = new ArrayList<>();
        for (Person person : persons) {
            nameList.add(person.getName());
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, nameList);
        personListView.setAdapter(adapter);

        // Setup SearchView listener
        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                return false;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                adapter.getFilter().filter(newText);
                return false;
            }
        });

        // Handle item clicks
        AlertDialog dialog = builder.create();
        personListView.setOnItemClickListener((parent, view, position, id) -> {
            String selectedName = adapter.getItem(position);
            new AlertDialog.Builder(this)
                    .setTitle("Confirm Delete")
                    .setMessage("Are you sure you want to delete " + selectedName + "?")
                    .setPositiveButton("Yes", (confirmDialog, which) -> {
                        deletePerson(selectedName); // Menghapus orang berdasarkan nama yang dipilih
                        dialog.dismiss();
                        confirmDialog.dismiss();
                    })
                    .setNegativeButton("No", (confirmDialog, which) -> {
                        confirmDialog.dismiss();
                    })
                    .show();
        });

        // Setup buttons
        Button btnCancel = dialogView.findViewById(R.id.btn_cancel);
        btnCancel.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }

    private void initializeModel() {
        try {
            tfliteModel = loadModelFile();

            // Inisialisasi GPU delegate jika tersedia
            Interpreter.Options options = new Interpreter.Options();
            try {
                GpuDelegate.Options delegateOptions = new GpuDelegate.Options();
                delegateOptions.setQuantizedModelsAllowed(true);
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
                Log.d(TAG, "GPU delegate digunakan");
            } catch (Exception e) {
                Log.w(TAG, "GPU delegate tidak tersedia, menggunakan CPU", e);
            }

            interpreter = new Interpreter(tfliteModel, options);

            int probabilityTensorIndex = 0;
            int[] probabilityShape = interpreter.getOutputTensor(probabilityTensorIndex).shape(); // {1, EMBEDDING_SIZE}
            DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

            tImage = new TensorImage(DataType.FLOAT32);
            tBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            Log.d(TAG, "Model loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error initializing model", e);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String recognize(float[] embedding) {
        if (!persons.isEmpty()) {
            ArrayList<Float> similarities = new ArrayList<>();
            for (Person person : persons) {
                similarities.add(cosineSimilarity(person.getEmbedding(), embedding));
            }
            float maxVal = similarities.stream().max(Float::compare).orElse(-1f);
            cleanUpExpiredLabels();
            if (maxVal > Float.parseFloat(confidenceValue)) {
                int index = similarities.indexOf(maxVal);
                if (index != -1) {
                    // Start a new thread to send the data to RabbitMQ
                    executorService.submit(() -> {
                        String name = persons.get(index).getName();
                        String nip = persons.get(index).getNip();
                        String encodedImage = bitmapToBase64(resizeBitmap(mBitmap,90,90));
                        if (sentLabels.containsKey(name) && !isExpired(sentLabels.get(name))) {
                            return;
                        }
                        try {
                            rabbitMQHelper.sendMessage(name, nip, encodedImage);
                            sentLabels.put(name, System.currentTimeMillis());
                        } catch (Exception e) {
                            Log.e(TAG, "Failed to send message to RabbitMQ", e);
                        }
                    });

                    return persons.get(index).getName();
                }
            }
        }
        return "unknown";
    }
    private Bitmap resizeBitmap(Bitmap bitmap, int maxWidth, int maxHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (width > maxWidth || height > maxHeight) {
            float scaleWidth = ((float) maxWidth) / width;
            float scaleHeight = ((float) maxHeight) / height;
            float scaleFactor = Math.min(scaleWidth, scaleHeight);

            width = Math.round(scaleFactor * width);
            height = Math.round(scaleFactor * height);

            bitmap = Bitmap.createScaledBitmap(bitmap, width, height, true);
        }

        return bitmap;
    }
    private void showToast(String message) {
        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(() -> Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show());
    }
    private boolean isExpired(long timestamp) {
        return System.currentTimeMillis() - timestamp > EXPIRATION_TIME;
    }
    private void cleanUpExpiredLabels() {
        Iterator<Map.Entry<String, Long>> iterator = sentLabels.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, Long> entry = iterator.next();
            if (isExpired(entry.getValue())) {
                iterator.remove(); // Hapus label yang telah kedaluwarsa
            }
        }
    }

    private String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
            return Base64.encodeToString(byteArray, Base64.NO_WRAP);
    }

    private void deletePerson(String name) {
        for (Person person : persons) {
            if (person.getName().equals(name)) {
                personDAO.deletePerson(person.getName()); // Menghapus dari database SQLite
                persons.remove(person); // Menghapus dari daftar orang yang ditampilkan
                Toast.makeText(MainActivity.this, "Person with Name " + name + " deleted", Toast.LENGTH_SHORT).show();
                break;
            }
        }
        // Memuat kembali semua orang setelah menghapus
        persons = personDAO.getAllPersons();
    }

    private float[] getEmbedding(Bitmap bitmap) {
        tImage.load(bitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(inputImageWidth, inputImageHeight, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();
        tImage = imageProcessor.process(tImage);

        interpreter.run(tImage.getBuffer(), tBuffer.getBuffer().rewind());
        return tBuffer.getFloatArray();
    }

    private float cosineSimilarity(float[] A, float[] B) {
        if (A == null || B == null || A.length == 0 || B.length == 0 || A.length != B.length) {
            return 2.0F;
        }

        double sumProduct = 0.0;
        double sumASq = 0.0;
        double sumBSq = 0.0;
        for (int i = 0; i < A.length; i++) {
            sumProduct += A[i] * B[i];
            sumASq += A[i] * A[i];
            sumBSq += B[i] * B[i];
        }
        if (sumASq == 0.0 && sumBSq == 0.0) {
            return 2.0F;
        } else {
            return (float) (sumProduct / (Math.sqrt(sumASq) * Math.sqrt(sumBSq)));
        }
    }

    private final BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    try {
                        InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }

                        inputStream.close();
                        os.close();

                        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (cascadeClassifier.empty()) {
                            Log.d(TAG, "Failed to load cascade classifier");
                        } else {
                            Log.d(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        }

                        cascadeDir.delete();
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    javaCameraView.setCameraIndex(ID_CAMERA);
                    javaCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBA = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRGBA.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRGBA = inputFrame.rgba();

        int orientation = javaCameraView.getScreenOrientation();
        if (javaCameraView.isEmulator()) {
            Core.flip(mRGBA, mRGBA, 1);
        } else {
            switch (orientation) {
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    if (ID_CAMERA == CameraBridgeViewBase.CAMERA_ID_BACK) Core.flip(mRGBA, mRGBA, 1);
                    Core.flip(mRGBA, mRGBA, 0);
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    if (ID_CAMERA == CameraBridgeViewBase.CAMERA_ID_BACK) Core.flip(mRGBA, mRGBA, 0);
                    Core.flip(mRGBA, mRGBA, 1);
                    break;
            }
        }

        Core.rotate(mRGBA, mRGBA, Core.ROTATE_90_COUNTERCLOCKWISE);

        MatOfRect detectedFaces = new MatOfRect();
        cascadeClassifier.detectMultiScale(mRGBA, detectedFaces);
        Rect[] rects = detectedFaces.toArray();

        if (rects.length == 0) {
            mBitmap = null;
        } else {
            for (Rect rect : rects) {
                Mat m = mRGBA.submat(rect);
                mBitmap = Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(m, mBitmap);
                String res = recognize(getEmbedding(mBitmap));
                Scalar scalar = res.equals("unknown") ? new Scalar(255, 0, 0) : new Scalar(0, 255, 0);

                Imgproc.rectangle(
                        mRGBA,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        scalar, 1
                );

                Imgproc.putText(
                        mRGBA,
                        res,
                        new Point(rect.x, rect.y - 5),
                        2,
                        0.5,
                        scalar, 2
                );
            }
        }

        Core.rotate(mRGBA, mRGBA, Core.ROTATE_90_CLOCKWISE);

        return mRGBA;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        javaCameraView.disableView();
    }

    @Override
    protected void onPause() {
        super.onPause();
        javaCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV is Configured or Connected Successfully");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCV is not working or Loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }
    public static class RabbitMQHelper {
        private final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        private ConnectionFactory factory;
        private Connection connection;
        private Channel channel;
        private MainActivity mainActivity;
        String rabbitmqHost = BuildConfig.RABBITMQ_HOST;
        int rabbitmqPort = BuildConfig.RABBITMQ_PORT;
        String rabbitmqUsername = BuildConfig.RABBITMQ_USERNAME;
        String rabbitmqPassword = BuildConfig.RABBITMQ_PASSWORD;
        String exchangeName = BuildConfig.EXCHANGE_NAME;
        String routingKey = BuildConfig.ROUTING_KEY;
        String siteCode = BuildConfig.SITE_CODE;

        public RabbitMQHelper(MainActivity mainActivity) {
            this.mainActivity = mainActivity;
            factory = new ConnectionFactory();
            factory.setUsername(rabbitmqUsername);
            factory.setPassword(rabbitmqPassword);
            factory.setHost(rabbitmqHost);
            factory.setPort(rabbitmqPort);
            factory.setAutomaticRecoveryEnabled(true);
        }

        public void connect() throws Exception {
                connection = factory.newConnection();
                channel = connection.createChannel();
                Log.d(TAG, "RabbitMQ connection established");

            // Add shutdown listener to handle reconnection
            connection.addShutdownListener(cause -> {
                if (cause.isInitiatedByApplication()) {
                    return;  // Ignore application-initiated shutdowns
                }
                Log.d("RabbitMQHelper", "Connection lost, reconnecting...");
                (mainActivity).connectWithRetry();  // Reconnect on shutdown
            });

        }

        public void sendMessage(String name,String nip, String encodedImage) throws JSONException, IOException {

            if (channel == null || !channel.isOpen()) {
                Log.d(TAG, "Channel is not open or null. Cannot send message.");
                return;
            }
                JSONObject json = new JSONObject();

                json.put("name", name);
                json.put("nip", nip);
                json.put("siteCode", siteCode);
                json.put("timestamp", LocalDateTime.now().format(FORMATTER));
                json.put("image", encodedImage);

                String message = json.toString();
                channel.basicPublish(exchangeName, routingKey, null, message.getBytes("UTF-8"));

                Handler handler = new Handler(Looper.getMainLooper());
                handler.post(() -> mainActivity.showToast(name + " absence successful."));


        }
        public void close() throws Exception {
            try {
                if (channel != null && channel.isOpen()) {
                    channel.close();
                }
                if (connection != null && connection.isOpen()) {
                    connection.close();
                }
                Log.d(TAG, "RabbitMQ connection closed");
            } catch (Exception e) {
                Log.d(TAG, "Error closing RabbitMQ connection");
            }
        }
    }
    private void connectWithRetry() {
        final int RETRY_DELAY_MS = 5000;  // Delay between retries in milliseconds

        new Thread(() -> {
            while (true) {
                try {
                    rabbitMQHelper = new RabbitMQHelper(MainActivity.this);
                    rabbitMQHelper.connect();
                    Log.d(TAG, "Connected to RabbitMQ");
                    break;  // Exit loop if connection is successful
                } catch (Exception e) {
                    Log.d(TAG, "Failed to connect to RabbitMQ, retrying in " + RETRY_DELAY_MS + " ms");
                    e.printStackTrace();
                    try {
                        Thread.sleep(RETRY_DELAY_MS);  // Wait before retrying
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();  // Restore interrupted status
                        break;
                    }
                }
            }
        }).start();
    }


    static {
        System.loadLibrary("native-lib");
    }
}
