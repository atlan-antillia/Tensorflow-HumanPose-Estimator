

<h2>Tensorflow-HumanPose-Estimator (Updated:2023/11/22)</h2>

<h3>1 Base Source Code </h3>
This is based on the following Tensorflow2 Pose Estimation.<br>

https://github.com/gsethi2409/tf-pose-estimation <br>

<h3>2 Development Environement</h3>
<h3>2.1 OS, Python and Tensorflow</h3> 
 We use the following development environment.<br>
<pre>
 Windows 11
 Python 3.10
 Tensorflow 2.15
</pre>

<h3>2.2 Python virual enviroment </h3>
Please create a python virtual env.<br>
<pre>
>python -m venv c:\py310-tfpose
</pre>

Activate the virtual environmet,<br>
<pre>
>cd c:\py310-tfpose
>source\activate
</pre>



<h3>2.3 Clone the repository</h3>

</h3>
<pre>
>mkdir c:\work
>cd c:\work
>git clone https://github.com/atlan-antillia/Tensorflow-HumanPose-Estimator.git
</pre>

<h3>2.4 Install python packages</h3>
<pre>
>cd Tensorflow-HumanPose-Estimator
>pip install -r requirments.txt
</pre>


<h3>3 Human Pose Estimation </h3>
<h3>3.1 Inference.config file </h3>
Please define your own inference.config file<br>
<pre>
[inference]
images_dir         = "./images"
outputs_dir        = "./outputs"
model              = "mobilenet_thin"
resize             = "656x368"
resize_out_ratio   = 4.0
debug              = False
</pre>

Please put your own human pose images under images_dir<br>
sample images<br>
<img src="./assets/images.png" width="1024" height="auto"><br>
 
<h3>3.2 Estimation </h3>
Please run the following command.<br>
<pre>
>python TensorflowHumanPoseEstimator.py
</pre>
sample outputs<br>
<img src="./assets/outputs.png" width="1024" height="auto"><br>

