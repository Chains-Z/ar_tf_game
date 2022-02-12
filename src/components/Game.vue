<template>
<div>
<iframe src="/web-desktop/index.html"
        height="640"
        width="960"
        ref="iframe"
></iframe>
<el-button @click="spaceDown">spaceDown</el-button>
  <el-button @click="spaceUp">spaceUp</el-button>
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