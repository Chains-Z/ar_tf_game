<template>
  <div class="home">
    <Game></Game>
    <Tf_Train></Tf_Train>
  </div>
</template>

<script>
import Game from "@/components/Game";
import Tf_Train from "@/components/Tf_Train";
export default {
  name: 'Home',
  components: {
    Game,Tf_Train
  }
}
</script>
