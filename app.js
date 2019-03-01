var express = require('express');
var fs = require("fs");
var bodyParser = require('body-parser');
var multer  = require('multer');
const path = require('path');
const spawn = require("child_process").spawn;

const port = process.env.PORT || 8081

var app = express();

app.use(express.static(__dirname+'/public'));

app.use(bodyParser.urlencoded({ extended: false }));

var upload = multer({ dest: '/tmp/'});

function testing(req, res){
    const pythonPro = spawn('python',["./main_test.py"]);
    pythonPro.stdout.on('data', (data) => {
        res.send(data.toString());
    });
}

app.get('/fileupload.html', function (req, res) {
   res.sendFile( __dirname + "/public/" + "index.html" );
})

app.post('/file_upload', upload.single('file'), function(req, res, next) {
 
    const tempPath = req.file.path;
    const targetPath = path.join(__dirname, "./uploads/image.jpg");
    fs.rename(tempPath, targetPath, function(err) {
      if (err) {
        console.log(err);
      }
    });
    return next();
  }, testing);

var server = app.listen(port, function () {
  var host = server.address().address
  console.log("Example app listening at http://%s:%s", host, port)
})