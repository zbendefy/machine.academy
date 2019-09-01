

class EvoDrawing
{
    constructor(canvas, imgName, resultsName, networkoutputName, offscreenCanvasName){
        this.canvas = canvas;
        this.context = canvas.getContext("2d");

        this.networkoutputName = networkoutputName;
        this.paused=false;
        this.resultsLabelName = resultsName;
        this.trackImageName = imgName;
        let img = document.getElementById(imgName);
        this.offscreenCanvas = document.getElementById(offscreenCanvasName);
        this.imgWidth = img.width;
        this.offscreenCanvas.width = img.width;
        this.offscreenCanvas.height = img.height;
        this.offscreenCanvas.getContext('2d').drawImage(img, 0, 0, img.width, img.height);
        this.offscreenCanvasContext = this.offscreenCanvas.getContext('2d');
        let offscreenCanvasImageRGBA = this.offscreenCanvasContext.getImageData(0, 0, img.width, img.height);
        this.offscreenCanvasImage = [];
        for(let iy = 0; iy < img.height; iy++){
            for(let ix = 0; ix < img.width; ix++){
                this.offscreenCanvasImage[iy * img.width + ix] = ( offscreenCanvasImageRGBA.data[(iy * img.width + ix)*4] ) > 0.2 ? true : false;
            }   
        }

        this.entities = [];

        this.entityCountTarget = 60;
        this.entityTimeoutS = 500;
        this.frameTimeS = 0.05; 
        this.generationSurvivorPercentage = 0.2;
        this.learningRate = 0.01;
        this.outlierChance = 0.02;
        this.outlierDelta = 20;

        this.drawFpsTarget = 45;
        this.drawFrameTimeTargetMs  = 1000 / this.drawFpsTarget;
        this.lastFrameTime = Date.now();

        this.currentSessionTimer = 0;
        this.currentGeneration = 1;
        this.simulationSpeed = 1;
        this.timerFnc = undefined;
        
        while( this.entities.length < this.entityCountTarget ){
            this.entities.push(new Entity(this.GetPixelAtPoint.bind(this)));
        }
        
        this._MutateEntities();

        this.SetSimulationSpeed(this.simulationSpeed);
    }

    _MutateEntities(skipFirst = false) {
        let elementCount = -1;
        for(let entity of this.entities){
            elementCount++;
            if (elementCount == 0 && skipFirst) 
                continue;
            entity.Mutate(this.learningRate);
        }
    }

    _Timeout() {
        for(let entity of this.entities){
            entity.GenerationEnd();
        }

        this.entities.sort( function(a,b){ return b.reward-a.reward; } ); //Sort by descending order

        if ( this.networkoutputName  != undefined ){
            let networkAsString = this.entities[0].brain.GetNetworkAsJSON();
            document.getElementById(this.networkoutputName).innerHTML = "//Generation " + this.currentGeneration + ", Reward: " + (this.entities[0].reward|0)
             +"\n\n" +  networkAsString;
        }

        this.entities = this.entities.slice(0, Math.floor(this.entities.length * this.generationSurvivorPercentage) );
        
        let survivorCount = this.entities.length;
        let currentEntityIdx = 0;
        while(this.entities.length < this.entityCountTarget){
            this.entities.push(this.entities[currentEntityIdx].DeepCopy());
            currentEntityIdx = (currentEntityIdx+1)%survivorCount;
        }

        this._MutateEntities(true);

        for(let entity of this.entities){
            entity.Reset();
        }
        
        this.currentSessionTimer = 0;
        this.currentGeneration++;
    }

    SetSurvivalRate(rate){ this.generationSurvivorPercentage = rate; }

    SetMutationRate(rate){ this.learningRate = rate; }

    SetSimulationSpeed(speed){
        this.simulationSpeed = speed;
        if (this.timerFnc)
            clearInterval(this.timerFnc);
        
        this.paused = this.simulationSpeed == 0;
		
        if ( this.simulationSpeed > 0 ){
            let tickTimeMs = (this.frameTimeS * 1000) / this.simulationSpeed;
            this.timerFnc = setInterval(()=>{this.Tick()}, tickTimeMs);
        } else {
            this.timerFnc = setInterval(()=>{this.Tick()}, 250);
		}
    }

    Simulate(dt) {
        if ( this.paused )
            return;

        let allEntitiesDisqualified = this.entities.every( (entity)=>{return entity.IsDisqualified()} );

        this.currentSessionTimer += dt;
        if ( this.currentSessionTimer > this.entityTimeoutS || allEntitiesDisqualified){
            this._Timeout();
        }

        for(let entity of this.entities){
            entity.Process(dt);
        }

    }

    Draw() {
        var img = document.getElementById(this.trackImageName);
        this.context.drawImage(img, 0, 0);
        
        for(let entity of this.entities){
            entity.Draw(this.context);
        }
        
        var resultsPanel = document.getElementById(this.resultsLabelName);

        resultsPanel.innerText = "Generation: " + this.currentGeneration + 
         "\nCars in race: " + this.entities.filter(x => !x.IsDisqualified()).length +
         "\nTimer: " + (this.currentSessionTimer|0) + "s";
    }

    Tick(){
        this.Simulate(this.frameTimeS);

        let currentTime = Date.now();
        if (currentTime - this.lastFrameTime >= this.drawFrameTimeTargetMs){
            this.Draw();
            this.lastFrameTime = currentTime;
        }
    }

    GetPixelAtPoint(x,y)
    {
        return this.offscreenCanvasImage[(y|0)*this.imgWidth+(x|0)];
    }
}

