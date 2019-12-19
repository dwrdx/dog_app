
<!DOCTYPE html>
<html>
<head>
<title>Funny Dog Breed Predictor</title>
<style>
img {
  width: 100%;
}
</style>
</head>
<body>

<br>
<br>
<h1> Funny Dog Breed Predictor </h1>
<h3> 1. Upload a dog image, predict the breed of the dog </h3>
<h3> 2. Upload a human image, predict the breed of the dog which the human looks like </h3>
<br>
<br>

<form action="/upload" method="post" enctype="multipart/form-data">
  Select a image file: <input type="file" name="upload" />
  <input type="submit" value="Start Predicting" />
</form>

<br>
<br>
<br>
<br>
<br>
<br>

<h3> Predicted breed : </h3>
<p>There is a {{human_or_dog}} in the image</p>
<p>The breed is {{dogname}}</p>
<img src="/static/{{filename}}" alt="HTML5 Icon" style="width:224px;height:224px;">


</body>
</html>