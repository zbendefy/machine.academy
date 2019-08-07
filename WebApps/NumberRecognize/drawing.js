
function loadFile(filePath) {
    var result = null;
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("GET", filePath, false);
    xmlhttp.send();
    if (xmlhttp.status==200) {
        result = xmlhttp.responseText;
    }
    return result;
}

var networkFilenames = [ 
    "https://raw.githubusercontent.com/zbendefy/machine.academy/master/WebApps/NumberRecognize/network1.json",
    "https://raw.githubusercontent.com/zbendefy/machine.academy/master/WebApps/NumberRecognize/network2.json",
    "https://raw.githubusercontent.com/zbendefy/machine.academy/master/WebApps/NumberRecognize/network3.json",
    "https://raw.githubusercontent.com/zbendefy/machine.academy/master/WebApps/NumberRecognize/network4.json"
    ];
var networks = [];

for (let value of networkFilenames) {
    let fileContent = loadFile(value);
    let network = new NeuralNetwork(JSON.parse(fileContent));
    networks.push(network);
    console.log("Loaded network: " + network.GetName());
}

class CanvasDrawing
{
    _moveEvent(clientX, clientY)
    {
        let canvas = this.canvas;
        let canvasSmall = this.canvasSmall;

        if ( this.canvasDrawing && clientX >= 0 && clientY >= 0 ) {
            let rect = canvas.getBoundingClientRect();
            let facx = (clientX - rect.left) / canvas.clientWidth;
            let facy = (clientY - rect.top) / canvas.clientHeight;

            let cs_x = Math.max(0, Math.min( facx * (canvasSmall.clientWidth-1), canvasSmall.clientWidth - 1));
            let cs_y = Math.max(0, Math.min( facy * (canvasSmall.clientHeight-1), canvasSmall.clientHeight - 1));

            let from_x = this.prevX < 0 ? cs_x : this.prevX;
            let from_y = this.prevY < 0 ? cs_y : this.prevY;

            let ctxSmall = canvasSmall.getContext("2d");
            ctxSmall.beginPath();
            ctxSmall.lineWidth = 1.6;
            ctxSmall.moveTo(from_x|0, from_y|0);
            ctxSmall.lineTo(cs_x|0, cs_y|0);
            ctxSmall.strokeStyle = "#000000";
            ctxSmall.stroke();
            ctxSmall.moveTo(cs_x|0, cs_y|0);
            ctxSmall.beginPath();

            this.prevX = cs_x;
            this.prevY = cs_y;

            let ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(canvasSmall, 0, 0, canvas.width, canvas.height);
        }
    }

    _startEvent(clientX, clientY){
        let canvas = this.canvas;
        let canvasSmall = this.canvasSmall;

        let rect = canvas.getBoundingClientRect();
        let facx = (clientX - rect.left) / canvas.clientWidth;
        let facy = (clientY - rect.top) / canvas.clientHeight;

        this.prevX= Math.max(0, Math.min( facx * (canvasSmall.clientWidth-1), canvasSmall.clientWidth - 1));
        this.prevY= Math.max(0, Math.min( facy * (canvasSmall.clientHeight-1), canvasSmall.clientHeight - 1));

        this.canvasDrawing = true;
    }
    
    _finishEvent(){
        this.canvasDrawing = false;
        this.prevX = -1;
        this.prevY = -1;
    }

    _getTouch(e) {
        let ret = [-1,-1];
        if (typeof e === "undefined")
            return ret;

        if (e.touches) {
            if (e.touches.length > 0) {
                var touch = e.touches[0];
                ret[0] = (touch.clientX);
                ret[1] = (touch.clientY);
            }
        }

        return ret;
    }

    constructor(canvasObj, smallCanvasObj)
    {
        'use strict'
        this.canvas = canvasObj;
        this.canvasSmall = smallCanvasObj;
        this.canvasDrawing = false;
        this.prevX = -1;
        this.prevY = -1;
        this.invertPixelValue = false;

        let canvas = this.canvas;
        let canvasSmall = this.canvasSmall;

        let drawing = this;

        canvas.addEventListener("mousemove", function (e) { drawing._moveEvent(e.clientX, e.clientY); }, false);
        canvas.addEventListener("mousedown", function (e) { drawing._startEvent(e.clientX, e.clientY); }, false);
        canvas.addEventListener("mouseup", function (e) { drawing._finishEvent(); }, false);
        canvas.addEventListener("mouseout", function (e) { drawing._finishEvent(); }, false);

        canvas.addEventListener("touchmove", function (e) {
              e.preventDefault();
              drawing._moveEvent(...drawing._getTouch(e)); 
            }, false);
        canvas.addEventListener("touchstart", function (e) {  
            e.preventDefault();
            drawing._startEvent(...drawing._getTouch(e)); 
        }, false);
        canvas.addEventListener("touchend", function (e) {  e.preventDefault(); drawing._finishEvent(); }, false);
        canvas.addEventListener("touchcancel", function (e) {  e.preventDefault(); drawing._finishEvent(); }, false);

        this.Clear();
    }

    GetContext()
    {
        return this.canvas.getContext("2d");
    }
    GetContextSmall()
    {
        return this.canvasSmall.getContext("2d");
    }
    
    Clear(){
        let ctx = this.GetContext();
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        let ctxSmall = this.GetContextSmall();
        ctxSmall.fillStyle = "#ffffff";
        ctxSmall.fillRect(0, 0, this.canvasSmall.width, this.canvasSmall.height);
    }

    DebugPrint(array)
    {
        let prt = "";
        for (let y = 0; y < 28; y++) { 
            for (let x = 0; x < 28; x++) { 
                if (y==0 || y==27 )
                    prt += "--";
                else if (x == 0  ||x==27)
                    prt += "|"
                else
                    prt += array[y*28+x] == 1 ? "  " : "##" ;
            }
            prt += "\n"
        }
        prt += "\n"
        console.log(prt);
    }

    _GetCenterOfMass(imgData)
    {
        let imgSizeX = 28;
        let imgSizeY = 28;
        let centerOfMassX = 0;
        let centerOfMassY = 0;
        
        let backgroundValue = this.invertPixelValue ? 0 : 1;
        let contributingPixels = 0;
        for (let y = 0; y < imgSizeY; y++) { 
            for (let x = 0; x < imgSizeX; x++) { 
                if ( imgData[y*imgSizeX + x] != backgroundValue ){
                    contributingPixels++;
                    centerOfMassX +=  x;
                    centerOfMassY +=  y;
                }
            }
        }

        centerOfMassX /=  contributingPixels;
        centerOfMassY /=  contributingPixels;

        return [centerOfMassX|0, centerOfMassY|0];
    }
    
    Calculate(target, centerAlign){
        let imgSizeX = 28;
        let imgSizeY = 28;

        let input = [];
        let result = [];
        
        for (let i = 0; i < 10; i++) { 
            result.push(0); 
        }
        
        let ctxSmall = this.GetContextSmall();
        let imgdata = ctxSmall.getImageData(0,0, this.canvasSmall.clientWidth, this.canvasSmall.clientHeight);
        for (let i = 0; i < this.canvasSmall.clientHeight; i++) { 
            for (let j = 0; j < this.canvasSmall.clientWidth; j++) { 
                let pxVal = imgdata.data[i * this.canvasSmall.clientWidth * 4 + j * 4];
                if (this.invertPixelValue)
                    pxVal = 255-pxVal;
                let inputActivation = pxVal / 255;
                input[i*this.canvasSmall.clientWidth+j] = inputActivation;
            }
        }
        
        if (centerAlign)  {
            let [centerOfMassX, centerOfMassY] = this._GetCenterOfMass(input);
            this._lastCenterOfMass = [centerOfMassX, centerOfMassY];

            let deltaX = Math.round((imgSizeX/2) - centerOfMassX)|0; 
            let deltaY = Math.round((imgSizeY/2) - centerOfMassY)|0;

            let translatedInput = [];

            for (let y = 0; y < imgSizeY; y++) { 
                for (let x = 0; x < imgSizeX; x++) { 
                    let sampleCoordX = x - deltaX;
                    let sampleCoordY = y - deltaY;
                    if (sampleCoordX < 0 || sampleCoordX >= imgSizeX 
                        || sampleCoordY < 0 || sampleCoordY >= imgSizeY )
                        {
                            translatedInput[y*imgSizeX + x] = this.invertPixelValue ? 0 : 1;
                        }
                        else
                        {
                            translatedInput[y*imgSizeX + x] = input[sampleCoordY*imgSizeX + sampleCoordX];
                        }
                }
            }

            //this.DebugPrint(input);
            //this.DebugPrint(translatedInput);

            input = translatedInput;

        }

        for (let n of networks) {
            let output = n.Calculate(input);
            console.log("Network output:" + output);
            let largestIdx = -1;
            let largestVal = -1;
            for (let i = 0; i < output.length; i++) { 
                if ( output[i] > largestVal ){
                    largestVal = output[i];
                    largestIdx = i;
                }
            }

            result[largestIdx]++;
        }

        let largestIdx = -1;
        let largestVal = -1;
        for (let i = 0; i < result.length; i++) { 
            if ( result[i] > largestVal ){
                largestVal = result[i];
                largestIdx = i;
            }
        }
        
        return [largestIdx, result];
    }
}

var networkCanvas = new CanvasDrawing(document.getElementById("drawCanvas"), document.getElementById("drawCanvasSmall"))
networkCanvas.invertPixelValue = true;

function clearImage()
{
    networkCanvas.Clear();
}

function calculateNetwork(target, centerOfMassTarget, detailTarget)
{
    let [result, voting] = networkCanvas.Calculate(target, true);
    document.getElementById(target).innerHTML = "I think that's a " + result+"!";

    let xCorrection = (14-networkCanvas._lastCenterOfMass[0]);
    let yCorrection = (14-networkCanvas._lastCenterOfMass[1]);
    let xSignPrefix = xCorrection > 0 ? "+" : "";
    let ySignPrefix = yCorrection > 0 ? "+" : "";

    document.getElementById(centerOfMassTarget).innerHTML =
     "Center of mass correction: (" + xSignPrefix + xCorrection + ", " + ySignPrefix + yCorrection+")";

    document.getElementById(detailTarget).innerHTML =
    "Voting result: [" + voting.toString() + "]";
}