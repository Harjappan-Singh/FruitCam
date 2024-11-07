package com.example.fruitzypredictor
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.example.fruitzypredictor.ml.Model
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    private val imageSize = 32
    private var result by mutableStateOf("Result will be displayed here")
    private var imageBitmap by mutableStateOf<Bitmap?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            var isSplashVisible by remember { mutableStateOf(true) }

            if (isSplashVisible) {
                com.example.fruitzypredictor.SplashScreen {
                    isSplashVisible = false
                }
            } else {
                MyApp(result, imageBitmap)
            }
        }


        requestPermissions()
    }

    @Composable
    fun MyApp(classificationResult: String, image: Bitmap?) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFFFFECB3)),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.frutzypredictorlogo),
                    contentDescription = null,
                    modifier = Modifier.size(48.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "FruitCam",
                    style = TextStyle(
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF143661)
                    )
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Display Image Preview
            Box(
                modifier = Modifier
                    .size(200.dp)  // Making the image preview larger
                    .background(Color.White, shape = RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                if (image != null) {
                    Image(
                        bitmap = image.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Text("Image", color = Color.Gray, fontSize = 18.sp)
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Result Text
            Text(
                text = "Result: $classificationResult",
                style = TextStyle(
                    fontSize = 18.sp,
                    color = Color(0xFF143661),
                    fontWeight = FontWeight.SemiBold
                )
            )

            Spacer(modifier = Modifier.height(24.dp))

            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Box(
                    modifier = Modifier
                        .background(Color(0xFF143661), shape = RoundedCornerShape(8.dp))
                        .clickable { openCamera() }
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Text("Open Camera", color = Color.White)
                }

                Box(
                    modifier = Modifier
                        .background(Color(0xFF143661), shape = RoundedCornerShape(8.dp))
                        .clickable { openGallery() }
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Text("Open Gallery", color = Color.White)
                }
            }
        }
    }

    private fun requestPermissions() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(cameraIntent)
    }

    private fun openGallery() {
        val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        galleryLauncher.launch(galleryIntent)
    }

    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val image = result.data?.extras?.get("data") as Bitmap
            val dimension = Math.min(image.width, image.height)
            val thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
            imageBitmap = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
            classifyImage(imageBitmap!!)
        }
    }

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val uri: Uri? = result.data?.data
            val image = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageBitmap = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
            classifyImage(imageBitmap!!)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = Model.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3).order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0

            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val `val` = intValues[pixel++]
                    byteBuffer.putFloat(((`val` shr 16) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat(((`val` shr 8) and 0xFF) * (1f / 1))
                    byteBuffer.putFloat((`val` and 0xFF) * (1f / 1  ))
                }
            }
            inputFeature0.loadBuffer(byteBuffer)
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            val confidences = outputFeature0.floatArray
            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
            val classes = arrayOf("Apple", "Banana", "Orange")

            val confidenceThreshold = 0.6f

            result = if (maxPos >= 0 && confidences[maxPos] >= confidenceThreshold) {
                classes[maxPos]
            } else {
                "Can't recognize"
            }

            model.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    @Preview(showBackground = true)
    @Composable
    fun DefaultPreview() {
        MyApp(result, imageBitmap)
    }
}

@Composable
fun SplashScreen(onSplashFinished: () -> Unit) {
    val splashImage = painterResource(id = R.drawable.fruitcamsplashscreen)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        Image(
            painter = splashImage,
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )

        CircularProgressIndicator(
            modifier = Modifier.align(Alignment.Center)
        )
    }

    LaunchedEffect (Unit) {
        delay(3000)
        onSplashFinished()
    }
}
