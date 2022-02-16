<template>
  <div>
<!--    游戏部分-->
    <el-row>
      <el-col>
        <iframe src="/web-desktop/index.html"
                height="640"
                width="960"
                ref="iframe"
        ></iframe>
      </el-col>
    </el-row>
<!--    tf训练部分-->
    <div ref="status">Loading mobilenet...</div>
    <el-row justify="center">
      <el-col :span="6">
        <el-row>
          <el-col :span="12">
            <el-button type="primary" size="large" @click="trainHandler">{{ train_status }}</el-button>
          </el-col>
          <el-col :span="12">
            <el-button type="primary" size="large" @click="playHandler">开始游戏</el-button>
          </el-col>
        </el-row>
        <el-row>
          <el-col :span="8">
            <el-form label-position="top">
              <el-form-item label="Learning rate">
                <el-select v-model="learning_rate" size="small">
                  <el-option :value="0.00001"></el-option>
                  <el-option :value="0.0001"></el-option>
                  <el-option :value="0.001"></el-option>
                  <el-option :value="0.003"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="Batch rate">
                <el-select v-model="batch_size" size="small">
                  <el-option :value="0.05"></el-option>
                  <el-option :value="0.1"></el-option>
                  <el-option :value="0.5"></el-option>
                  <el-option :value="1"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="Epochs">
                <el-select v-model="epochs" size="small">
                  <el-option :value="10"></el-option>
                  <el-option :value="20"></el-option>
                  <el-option :value="40"></el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="Hidden units">
                <el-select v-model="hidden_units" size="small">
                  <el-option :value="10"></el-option>
                  <el-option :value="100"></el-option>
                  <el-option :value="200"></el-option>
                </el-select>
              </el-form-item>
            </el-form>
          </el-col>
          <el-col :span="16">
            <div class="webcam-box-outer">
              <div class="webcam-box-inner">
                <video autoplay playsinline muted id="webcam" width="224" height="224"></video>
              </div>
            </div>
          </el-col>
        </el-row>
      </el-col>
      <el-col :span="6">
        <el-row style="margin-bottom: 20px;">
          点击以将当前摄像头画面作为一个Example添加给该游戏按键
        </el-row>
        <el-row style="text-align: center">
          <el-col :span="12">
            <el-row>
              <el-col>蓄力</el-col>
            </el-row>
            <el-row>
              <el-col>
                <div class="thumb-box-outer charge-thumb">
                  <div class="thumb-box-inner">
                    <canvas width=224 height=224
                            ref="charge-thumb" @click="addExamples(0)"
                            class="thumb"></canvas>
                  </div>
                </div>
              </el-col>
            </el-row>
            <el-row>
              <el-col>{{ charge_examples }} Examples</el-col>
            </el-row>
          </el-col>
          <el-col :span="12">
            <el-row>
              <el-col>跳</el-col>
            </el-row>
            <el-row>
              <el-col>
                <div class="thumb-box-outer jump-thumb">
                  <div class="thumb-box-inner">
                    <canvas width=224 height=224
                            ref="jump-thumb" @click="addExamples(1)"
                            class="thumb"></canvas>
                  </div>
                </div>
              </el-col>
            </el-row>
            <el-row>
              <el-col>{{ jump_examples }} Examples</el-col>
            </el-row>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import {ControllerDataset} from '@/assets/controller_dataset';

const NUM_CLASSES = 2//跳和蓄力
let truncatedMobileNet
let model
let webcam
let isPredicting = false
const CONTROLS = ['charge', 'jump']
let lastPredictId = 1
const controllerDataset = new ControllerDataset(NUM_CLASSES)
export default {
  name: "Tf_Train",
  data() {
    return {
      learning_rate: 0.0001,
      batch_size: 0.5,
      epochs: 20,
      hidden_units: 100,
      train_status: "训练模型",
      charge_examples: 0,
      jump_examples: 0,
    }
  },
  mounted() {
    this.init()
  },
  methods: {
    spaceDown() {
      this.$refs.iframe.contentWindow.postMessage('spaceDown', '*')
    },
    spaceUp() {
      this.$refs.iframe.contentWindow.postMessage('spaceUp', '*')
    },
    async addExamples(label) {
      const count_name = CONTROLS[label] + '_examples'
      this[count_name]++
      let img = await this.getImage();
      controllerDataset.addExample(truncatedMobileNet.predict(img), label);
      // Draw the preview thumbnail.
      this.drawThumb(img, label);
      img.dispose();
    },
    drawThumb(img, label) {
      const thumbCanvas = this.$refs[CONTROLS[label] + '-thumb']
      this.draw(img, thumbCanvas)
    },
    draw(image, canvas) {
      const [width, height] = [224, 224];
      const ctx = canvas.getContext('2d');
      const imageData = new ImageData(width, height);
      const data = image.dataSync();
      for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j] = (data[i * 3] + 1) * 127;
        imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
        imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
        imageData.data[j + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    },
    async loadTruncatedMobileNet() {
      const mobilenet = await tf.loadLayersModel(
          'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

      // Return a model that outputs an internal activation.
      const layer = mobilenet.getLayer('conv_pw_13_relu');
      return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    },
    async train() {

      if (controllerDataset.xs == null) {
        throw new Error('Add some examples before training!');
      }

      // Creates a 2-layer fully connected model. By creating a separate model,
      // rather than adding layers to the mobilenet model, we "freeze" the weights
      // of the mobilenet model, and only train weights from the new model.
      model = tf.sequential({
        layers: [
          // Flattens the input to a vector so we can use it in a dense layer. While
          // technically a layer, this only performs a reshape (and has no training
          // parameters).
          tf.layers.flatten(
              {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
          // Layer 1.
          tf.layers.dense({
            units: this.hidden_units,
            activation: 'relu',
            kernelInitializer: 'varianceScaling',
            useBias: true
          }),
          // Layer 2. The number of units of the last layer should correspond
          // to the number of classes we want to predict.
          tf.layers.dense({
            units: NUM_CLASSES,
            kernelInitializer: 'varianceScaling',
            useBias: false,
            activation: 'softmax'
          })
        ]
      });
      // Creates the optimizers which drives training of the model.
      const optimizer = tf.train.adam(this.learning_rate);
      // We use categoricalCrossentropy which is the loss function we use for
      // categorical classification which measures the error between our predicted
      // probability distribution over classes (probability that an input is of each
      // class), versus the label (100% probability in the true class)>
      model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

      // We parameterize batch size as a fraction of the entire dataset because the
      // number of examples that are collected depends on how many examples the user
      // collects. This allows us to have a flexible batch size.
      const batchSize =
          Math.floor(controllerDataset.xs.shape[0] * this.batch_size);
      console.log(this.batch_size)
      console.log(controllerDataset.xs.shape[0])
      console.log(batchSize)
      if (!(batchSize > 0)) {
        throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
      }

      // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
      await model.fit(controllerDataset.xs, controllerDataset.ys, {
        batchSize,
        epochs: this.epochs,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            this.train_status = 'Loss: ' + logs.loss.toFixed(5)
          }
        }
      });
    },
    async predict() {
      while (isPredicting) {
        // Capture the frame from the webcam.
        const img = await this.getImage();

        // Make a prediction through mobilenet, getting the internal activation of
        // the mobilenet model, i.e., "embeddings" of the input images.
        const embeddings = truncatedMobileNet.predict(img);

        // Make a prediction through our newly-trained model using the embeddings
        // from mobilenet as input.
        const predictions = model.predict(embeddings);

        // Returns the index with the maximum probability. This number corresponds
        // to the class the model thinks is the most probable given the input.
        const predictedClass = predictions.as1D().argMax();
        const classId = (await predictedClass.data())[0];
        img.dispose();

        this.predictClass(classId);
        await tf.nextFrame();
      }
    },
    predictClass(classId) {
      if(classId !== lastPredictId){
        lastPredictId = classId
        if(classId === 0)
          this.spaceDown()
        else
          this.spaceUp()
      }
      document.body.setAttribute('data-active', CONTROLS[classId])
    },
    async trainHandler() {
      this.train_status = 'Training...';
      await tf.nextFrame();
      await tf.nextFrame();
      isPredicting = false;
      await this.train();
    },
    async playHandler() {
      isPredicting = true;
      await this.predict();
    },
    async getImage() {
      const img = await webcam.capture();
      const processedImg =
          tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
      img.dispose();
      return processedImg;
    },
    async init() {
      try {
        webcam = await tfd.webcam(document.getElementById('webcam'));
      } catch (e) {
        console.log(e);
      }
      truncatedMobileNet = await this.loadTruncatedMobileNet();
      this.$refs.status.style.display = "none"

      // Warm up the model. This uploads weights to the GPU and compiles the WebGL
      // programs so the first time we collect data from the webcam it will be
      // quick.
      const screenShot = await webcam.capture();
      truncatedMobileNet.predict(screenShot.expandDims(0));
      screenShot.dispose();
    }
  }
}


</script>

<style scoped>
.webcam-box-outer {
  border: 1px solid #585858;
  border-radius: 4px;
  box-sizing: border-box;
  display: inline-block;
  padding: 9px;
  margin: 10px 0;
}

.webcam-box-inner {
  border: 1px solid #585858;
  border-radius: 4px;
  box-sizing: border-box;
  display: flex;
  justify-content: center;
  overflow: hidden;
  /*width: 160px;*/
}

#webcam {
  /*height: 160px;*/
  transform: scaleX(-1);
}

.thumb {
  /*height: 66px;*/
  height: 160px;
  transform: scaleX(-1);
}

.thumb-box-outer {
  border: 1px solid #585858;
  border-radius: 4px;
  box-sizing: border-box;
  display: inline-block;
  padding: 9px;
  position: relative;
  transition: box-shadow 0.3s;
  margin: 10px 0;
}

.thumb-box-inner {
  border: 1px solid #585858;
  border-radius: 4px;
  box-sizing: border-box;
  display: flex;
  justify-content: center;
  overflow: hidden;
}

[data-active="charge"] .charge-thumb,
[data-active="jump"] .jump-thumb {
  box-shadow: 0 0 6px 6px #42b983;
}
</style>