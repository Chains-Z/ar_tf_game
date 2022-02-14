<template>
<div>

  <el-row>
    <el-col>
      <iframe src="/web-desktop/index.html"
              height="640"
              width="960"
              ref="iframe"
      ></iframe>
    </el-col>
  </el-row>
<el-row>
  <el-col :offset="6" :span="4">
    <el-button @click="spaceDown">spaceDown</el-button>
    <el-button @click="spaceUp">spaceUp</el-button>
  </el-col>
</el-row>
</div>
</template>

<script>
export default {
  name: "Game",
  methods:{
    spaceDown(){
      this.$refs.iframe.contentWindow.postMessage('spaceDown','*')
    },
    spaceUp(){
      this.$refs.iframe.contentWindow.postMessage('spaceUp','*')
    }
  }
}
</script>

<style scoped>

</style>