package com.myapplication.alpr

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
//import com.google.firebase.firestore.FirebaseFirestore
import io.fotoapparat.Fotoapparat
import io.fotoapparat.preview.Frame
import io.fotoapparat.preview.FrameProcessor
import io.fotoapparat.selector.front
import io.fotoapparat.selector.back
import io.fotoapparat.view.CameraView
import org.buyun.alpr.sdk.SDK_IMAGE_TYPE
import org.buyun.alpr.sdk.AlprSdk
import org.buyun.alpr.sdk.AlprCallback
import org.buyun.alpr.sdk.AlprResult
import java.nio.ByteBuffer

import android.media.ExifInterface
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.Toast
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import java.text.SimpleDateFormat
import java.util.Date

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

class CameraActivityKt : AppCompatActivity() {

    val TAG = "KBY-AI ALPR"
    val PREVIEW_WIDTH = 720
    val PREVIEW_HEIGHT = 1280

    private lateinit var cameraView: CameraView
    private lateinit var faceView: FaceView
    private lateinit var fotoapparat: Fotoapparat
    private lateinit var context: Context

    private var recognized = false

    private lateinit var yoloInterpreter: Interpreter
    private lateinit var cnnInterpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_kt)

        context = this
        cameraView = findViewById(R.id.preview)
        faceView = findViewById(R.id.faceView)

//        if (SettingsActivity.getCameraLens(context) == CameraSelector.LENS_FACING_BACK) {
//            fotoapparat = Fotoapparat.with(this)
//                .into(cameraView)
//                .lensPosition(back())
//                .frameProcessor(FaceFrameProcessor())
//                .build()
//        } else  {
//            fotoapparat = Fotoapparat.with(this)
//                .into(cameraView)
//                .lensPosition(front())
//                .frameProcessor(FaceFrameProcessor())
//                .build()
//        }

//        Fotoapparat = library untuk preview dan menangani kamera.
        fotoapparat = Fotoapparat.with(this)
//            ditampilkan ke CameraView di layout
            .into(cameraView)
//            pakai kamera belakang
            .lensPosition(back())
//            proses setiap frame melalui FaceFrameProcessor
            .frameProcessor(FaceFrameProcessor())
            .build()



        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_DENIED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        } else {
            fotoapparat.start()
        }


//Interpreter dari TensorFlow Lite untuk menjalankan model .tflite
        yoloInterpreter = Interpreter(FileUtil.loadMappedFile(this, "yolov8.tflite"))
        cnnInterpreter = Interpreter(FileUtil.loadMappedFile(this, "cnn.tflite"))

    }

//    Menghentikan dan memulai kamera saat Activity dijeda/lanjut.
    override fun onResume() {
        super.onResume()
        recognized = false
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            fotoapparat.start()
        }
    }

    override fun onPause() {
        super.onPause()
        fotoapparat.stop()
        faceView.setFaceBoxes(null)
    }

//    Mengatur permission kamera.
    override fun onRequestPermissionsResult(
        requestCode: Int,
//        permissions: Array<String?>,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                fotoapparat.start()
            }
        }
    }

    inner class FaceFrameProcessor : FrameProcessor {

        override fun process(frame: Frame) {
//Pencegahan Deteksi Berulang
//            Mencegah agar satu plat nomor tidak diproses berulang-ulang selama BottomSheet belum ditutup.
//            Setelah BottomSheet ditutup → recognized = false agar bisa mendeteksi plat lagi.
            if(recognized == true) {
                return
            }

            val exifOrientation: Int = when (frame.rotation) {
                90 -> ExifInterface.ORIENTATION_ROTATE_270
                180 -> ExifInterface.ORIENTATION_ROTATE_180
                270 -> ExifInterface.ORIENTATION_ROTATE_90
                else -> ExifInterface.ORIENTATION_NORMAL
            }

            val bitmap = AlprSdk.yuv2Bitmap(frame.image, frame.size.width, frame.size.height, exifOrientation)

            val widthInBytes = bitmap.rowBytes
            val width = bitmap.width
            val height = bitmap.height
            val nativeBuffer = ByteBuffer.allocateDirect(widthInBytes * height)
            bitmap.copyPixelsToBuffer(nativeBuffer)
            nativeBuffer.rewind()

//            Mengubah frame kamera ke format bitmap → buffer → diproses oleh ALPR SDK
            val alprResult: AlprResult = AlprSdk.process(
                SDK_IMAGE_TYPE.ULTALPR_SDK_IMAGE_TYPE_RGBA32,
                nativeBuffer, width.toLong(), height.toLong()
            )
//            plates menyimpan hasil deteksi plat kendaraan dari gambar.
            val plates = AlprUtils.extractPlates(alprResult);

            val platesMap: ArrayList<HashMap<String, Any>> = ArrayList<HashMap<String, Any>>()

            runOnUiThread {

                if(!plates.isNullOrEmpty()) {
                    for(plate in plates) {
//                        Log.i("alprEngine", "number: " + plate.getNumber())
//                        Log.i("alprEngine", "wrapper: " + plate.getWarpedBox()[0])
                        val e: HashMap<String, Any> = HashMap<String, Any>()

                        var x1 = 65536.0f
                        var y1 = 65536.0f
                        var x2 = 0.0f
                        var y2 = 0.0f
                        val wrapper = plate.getWarpedBox()
                        if(wrapper[0] < x1) {
                            x1 = wrapper[0]
                        }
                        if(wrapper[1 * 2] < x1) {
                            x1 = wrapper[1 * 2]
                        }
                        if(wrapper[2 * 2] < x1) {
                            x1 = wrapper[2 * 2]
                        }
                        if(wrapper[3 * 2] < x1) {
                            x1 = wrapper[3 * 2]
                        }

                        if(wrapper[0 * 2 + 1] < y1) {
                            y1 = wrapper[0 * 2 + 1]
                        }
                        if(wrapper[1 * 2 + 1] < y1) {
                            y1 = wrapper[1 * 2 + 1]
                        }
                        if(wrapper[2 * 2 + 1] < y1) {
                            y1 = wrapper[2 * 2 + 1]
                        }
                        if(wrapper[3 * 2 + 1] < y1) {
                            y1 = wrapper[3 * 2 + 1]
                        }

                        if(wrapper[0 * 2] > x2) {
                            x2 = wrapper[0 * 2]
                        }
                        if(wrapper[1 * 2] > x2) {
                            x2 = wrapper[1 * 2]
                        }
                        if(wrapper[2 * 2] > x2) {
                            x2 = wrapper[2 * 2]
                        }
                        if(wrapper[3 * 2] > x2) {
                            x2 = wrapper[3 * 2]
                        }

                        if(wrapper[0 * 2 + 1] > y2) {
                            y2 = wrapper[0 * 2 + 1]
                        }
                        if(wrapper[1 * 2 + 1] > y2) {
                            y2 = wrapper[1 * 2 + 1]
                        }
                        if(wrapper[2 * 2 + 1] > y2) {
                            y2 = wrapper[2 * 2 + 1]
                        }
                        if(wrapper[3 * 2 + 1] > y2) {
                            y2 = wrapper[3 * 2 + 1]
                        }

                        e.put("x1", x1);
                        e.put("y1", y1);
                        e.put("x2", x2);
                        e.put("y2", y2);
                        e.put("frameWidth", bitmap!!.width);
                        e.put("frameHeight", bitmap!!.height);
                        e.put("number", plate.getNumber());
                        platesMap.add(e)

//                        val detectedPlate = plate.getNumber() // Hasil plat nomor yang terbaca
                        val detectedPlate = plate.getNumber().trim() // Membersihkan whitespace
                        Log.d(TAG, "Detected Plate: $detectedPlate")

                        // PERBAIKAN 1: Set recognized = true untuk menghentikan proses lain
                        //setelah plat terdeteksi untuk mencegah proses ganda
                        recognized = true

                        // Panggil fungsi untuk mengambil data dari Firebase
                        fetchVehicleDataFromFirebase(detectedPlate)

                    }
                }

                faceView.setFrameSize(Size(bitmap.width, bitmap.height))
                faceView.setFaceBoxes(platesMap)

            }
        }
    }

    // Fungsi untuk mengambil data dari Firebase berdasarkan plat nomor
    private fun fetchVehicleDataFromFirebase(plateNumber: String) {

        // PERBAIKAN 2: Log untuk debugging
        Log.d(TAG, "Mencoba mengambil data untuk plat: $plateNumber")

//        Mengambil Data Kendaraan dari Firebase Realtime Database
//        Plat adalah node utama di Firebase tempat data kendaraan disimpan.
//        Plat nomor yang terdeteksi dijadikan key untuk mencari informasi kendaraan.
        val db = FirebaseDatabase.getInstance().getReference("Plat")

        // PERBAIKAN 3: Tambahkan listener untuk status koneksi Firebase
        val connectedRef = FirebaseDatabase.getInstance().getReference(".info/connected")
        connectedRef.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                val connected = snapshot.getValue(Boolean::class.java) ?: false
                if (connected) {
                    Log.d(TAG, "Terhubung ke Firebase")
                } else {
                    Log.d(TAG, "Tidak terhubung ke Firebase")
                    // PERBAIKAN 4: Tampilkan pesan error koneksi
                    runOnUiThread {
//                        Toast.makeText(this@YourActivityName, "Tidak dapat terhubung ke database", Toast.LENGTH_SHORT).show()
                        Toast.makeText(this@CameraActivityKt, "Tidak dapat terhubung ke database", Toast.LENGTH_SHORT).show()
                    }
                }
            }

            override fun onCancelled(error: DatabaseError) {
                Log.e(TAG, "Listener dibatalkan", error.toException())
            }
        })

        // PERBAIKAN 5: Coba beberapa format plat
        val platFormats = arrayOf(
            plateNumber,                     // Format asli
            plateNumber.replace(" ", ""),    // Tanpa spasi
            plateNumber.toUpperCase(),       // Semua uppercase
            plateNumber.toLowerCase()        // Semua lowercase
        )

        var dataFound = false

        // PERBAIKAN 6: Tambahkan penanganan timeout
        val handler = Handler(Looper.getMainLooper())
        val timeoutRunnable = Runnable {
            if (!dataFound) {
                Log.d(TAG, "Timeout - data tidak ditemukan dalam waktu yang ditentukan")
                runOnUiThread {
//                    Toast.makeText(this@YourActivityName, "Data plat nomor tidak ditemukan", Toast.LENGTH_SHORT).show()
                    Toast.makeText(this@CameraActivityKt, "Data plat nomor tidak ditemukan", Toast.LENGTH_SHORT).show()
                }
            }
        }
//        Menangani Timeout & Koneksi
//        Jika dalam 10 detik data tidak ditemukan → tampilkan Toast bahwa data tidak ditemukan.
//        Juga ada pengecekan koneksi ke Firebase Realtime Database: .info/connected.
        handler.postDelayed(timeoutRunnable, 10000) // 10 detik timeout

        // PERBAIKAN 7: Coba dengan beberapa format plat
        for (format in platFormats) {
            db.child(format).get().addOnSuccessListener { snapshot ->
                // PERBAIKAN 8: Batalkan handler timeout jika data ditemukan
//                if (snapshot.exists()) {
                if (snapshot.exists() && !dataFound) {
                    dataFound = true
                    handler.removeCallbacks(timeoutRunnable)

//                    val owner = snapshot.child("Nama Pemilik").getValue(String::class.java) ?: "Tidak Diketahui"
                    val owner = snapshot.child("Nama Pemilik").getValue(String::class.java)
                        ?: snapshot.child("nama pemilik").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("NAMA PEMILIK").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val jenis = snapshot.child("Jenis Kendaraan").getValue(String::class.java)
                        ?: snapshot.child("jenis kendaraan").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("JENIS KENDARAAN").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val model = snapshot.child("Model").getValue(String::class.java)
                        ?: snapshot.child("model").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("MODEL").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val warna = snapshot.child("Warna Kendaraan").getValue(String::class.java)
                        ?: snapshot.child("warna kendaraan").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("WARNA KENDARAAN").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

//                    val year = snapshot.child("Tahun").getValue(String::class.java) ?: "Tidak Diketahui"
                    //val year = snapshot.child("Tahun").getValue(Long::class.java) ?: 0L
                    //val year = snapshot.child("Tahun").getValue(Long::class.java)?.toString() ?: "Tidak Diketahui"
                    val year = snapshot.child("Tahun Buat").getValue(Long::class.java)?.toString()
                        ?: snapshot.child("tahun buat").getValue(Long::class.java)?.toString()
                        ?: snapshot.child("TAHUN BUAT").getValue(Long::class.java)?.toString()
                        ?: "Tidak Diketahui"

                    val rangka = snapshot.child("No Rangka").getValue(String::class.java)
                        ?: snapshot.child("no rangka").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("NO RANGKA").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val mesin = snapshot.child("No Mesin").getValue(String::class.java)
                        ?: snapshot.child("no mesin").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("NO MESIN").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val silinder = snapshot.child("Isi Silinder").getValue(String::class.java)
                        ?: snapshot.child("isi silinder").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("ISI SILINDER").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val bahanbakar = snapshot.child("Bahan Bakar").getValue(String::class.java)
                        ?: snapshot.child("bahan bakar").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("BAHAN BAKAR").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val stnk = snapshot.child("STNK Berlaku").getValue(String::class.java)
                        ?: snapshot.child("stnk berlaku").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("STNK BERLAKU").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val akhirbayar = snapshot.child("Tanggal Akhir Bayar").getValue(String::class.java)
                        ?: snapshot.child("tanggal akhir bayar").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("TANGGAL AKHIR BAYAR").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val jatuhtempo = snapshot.child("Tanggal Jatuh Tempo").getValue(String::class.java)
                        ?: snapshot.child("tanggal jatuh tempo").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("TANGGAL JATUH TEMPO").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val pkb = snapshot.child("PKB").getValue(String::class.java)
                        ?: snapshot.child("pkb").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("PKB").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val denda = snapshot.child("Denda").getValue(String::class.java)
                        ?: snapshot.child("denda").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("DENDA").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

                    val jumlah = snapshot.child("Jumlah").getValue(String::class.java)
                        ?: snapshot.child("jumah").getValue(String::class.java)  // Coba dengan lowercase
                        ?: snapshot.child("JUMLAH").getValue(String::class.java)  // Coba dengan uppercase
                        ?: "Tidak Diketahui"

//                    Log.d(TAG, "Data ditemukan: $owner, $model, $year")
                    Log.d(TAG, "Data ditemukan untuk plat $format: $owner, $jenis, $model, $warna, $year, $rangka, $mesin, $silinder, $bahanbakar, $stnk, $akhirbayar, $jatuhtempo, $pkb, $denda, $jumlah")

                    // PERBAIKAN 9: Gunakan handler UI untuk memastikan UI diperbarui di thread utama
                    runOnUiThread {
//                        showBottomSheet(plateNumber, owner, model, year)
                        showBottomSheet(format, owner, jenis, model, warna, year, rangka, mesin, silinder, bahanbakar, stnk, akhirbayar, jatuhtempo, pkb, denda, jumlah)

                    }
                } else if (!dataFound) {
                    Log.d(TAG, "Data tidak ditemukan untuk format plat: $format")
                }
            }.addOnFailureListener { e ->
                Log.e(TAG, "Gagal mengambil data dari Realtime Database untuk format $format", e)

                // PERBAIKAN 10: Tampilkan pesan error yang lebih informatif
                if (!dataFound) {
                    runOnUiThread {
//                        Toast.makeText(this@YourActivityName, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                        Toast.makeText(this@CameraActivityKt, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }

        // PERBAIKAN 11: Debug struktur database
        db.addListenerForSingleValueEvent(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                Log.d(TAG, "Database struktur - children count: ${snapshot.childrenCount}")
                for (child in snapshot.children) {
                    Log.d(TAG, "Child key: ${child.key}")
                }
            }

            override fun onCancelled(error: DatabaseError) {
                Log.e(TAG, "Database browse error", error.toException())
            }
        })
    }

    // Fungsi untuk menampilkan hasil dalam Bottom Sheet
    @SuppressLint("MissingInflatedId")
    private fun showBottomSheet(plate: String, owner: String, jenis: String, model: String, warna: String, year: String, rangka: String, mesin: String, silinder: String, bahanbakar: String, stnk: String, akhirbayar: String, jatuhtempo: String, pkb: String, denda: String, jumlah: String) {
        // PERBAIKAN 12: Cek jika Activity masih aktif untuk menghindari crash
        if (isFinishing || isDestroyed) {
            Log.d(TAG, "Activity sudah tidak aktif, tidak dapat menampilkan bottom sheet")
            return
        }

//        Menampilkan Informasi di Bottom Sheet
//        Membuat dan menampilkan BottomSheetDialog untuk menampilkan data kendaraan.
//        Setiap TextView seperti plateTextView, ownerTextView diisi dengan data yang diambil dari Firebase.
        try {
            val bottomSheet = BottomSheetDialog(this)
            val view = layoutInflater.inflate(R.layout.bottom_sheet_layout, null)

//            val plateTextView = view.findViewById<TextView>(R.id.plateTextView)
            val plateTextView = view.findViewById<TextView>(R.id.plateTextView) ?: return
//            val ownerTextView = view.findViewById<TextView>(R.id.ownerTextView)
            val ownerTextView = view.findViewById<TextView>(R.id.ownerTextView) ?: return
//            val modelTextView = view.findViewById<TextView>(R.id.modelTextView)
//            val modelTextView = view.findViewById<TextView>(R.id.modelTextView) ?: return
            val jenisTextView = view.findViewById<TextView>(R.id.jenisTextView) ?: return
            val modelTextView = view.findViewById<TextView>(R.id.merkTextView) ?: return
            val warnaTextView = view.findViewById<TextView>(R.id.warnaTextView) ?: return
//            val yearTextView = view.findViewById<TextView>(R.id.yearTextView)
            val yearTextView = view.findViewById<TextView>(R.id.yearTextView) ?: return
            val rangkaTextView = view.findViewById<TextView>(R.id.rangkaTextView) ?: return
            val mesinTextView = view.findViewById<TextView>(R.id.mesinTextView) ?: return
            val silinderTextView = view.findViewById<TextView>(R.id.silinderTextView) ?: return
            val bahanbakarTextView = view.findViewById<TextView>(R.id.bahanbakarTextView) ?: return
            val stnkTextView = view.findViewById<TextView>(R.id.stnkTextView) ?: return
            val akhirbayarTextView = view.findViewById<TextView>(R.id.akhirbayarTextView) ?: return
            val jatuhtempoTextView = view.findViewById<TextView>(R.id.jatuhtempoTextView) ?: return
            val pkbTextView = view.findViewById<TextView>(R.id.pkbTextView) ?: return
            val dendaTextView = view.findViewById<TextView>(R.id.dendaTextView) ?: return
            val jumlahTextView = view.findViewById<TextView>(R.id.jumlahTextView) ?: return

            plateTextView.text = "Plat: $plate"
            ownerTextView.text = "Nama Pemilik: $owner"
            jenisTextView.text = "Jenis Kendaraan: $jenis"
            modelTextView.text = "Merk / Tipe: $model"
            warnaTextView.text = "Warna Kendaraan: $warna"
            yearTextView.text = "Tahun: $year"
            rangkaTextView.text = "No Rangka: $rangka"
            mesinTextView.text = "No Mesin: $mesin"
            silinderTextView.text = "Isi Silinder: $silinder"
            bahanbakarTextView.text = "Bahan Bakar: $bahanbakar"
            stnkTextView.text = "STNK Berlaku: $stnk"
            akhirbayarTextView.text = "Tanggal Akhir Bayar: $akhirbayar"
            jatuhtempoTextView.text = "Tanggal Jatuh Tempo: $jatuhtempo"
            pkbTextView.text = "Pajak: $pkb"
            dendaTextView.text = "Denda: $denda"
            jumlahTextView.text = "Jumlah: $jumlah"
//            modelTextView.text = "Model: $model"

            bottomSheet.setContentView(view)

            // PERBAIKAN 13: Tambahkan listener untuk bottom sheet
            bottomSheet.setOnDismissListener {
                Log.d(TAG, "Bottom sheet ditutup")
                // PERBAIKAN 14: Atur recognized = false lagi untuk mengaktifkan kembali pendeteksian
                recognized = false
            }

//            // PERBAIKAN 15: Tambahkan tombol selesai di bottom sheet
//            val doneButton = view.findViewById<Button>(R.id.doneButton)
//            doneButton?.setOnClickListener {
//                bottomSheet.dismiss()
//            }

            // Log untuk konfirmasi
            Log.d(TAG, "Menampilkan bottom sheet dengan data plat: $plate")

            bottomSheet.show()
        } catch (e: Exception) {
            Log.e(TAG, "Error saat menampilkan bottom sheet", e)
            Toast.makeText(this, "Terjadi kesalahan: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}