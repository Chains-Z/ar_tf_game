import { createStore } from 'vuex'

// 创建一个新的 store 实例
const store = createStore({
    state () {
        return {
            count: 0
        }
    },
    mutations: {
        setCount (state,n) {
            state.count = n
        }
    }
})
export default store