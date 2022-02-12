const AutoImport = require('unplugin-auto-import/webpack')
const Components = require('unplugin-vue-components/webpack')
const { ElementPlusResolver } = require('unplugin-vue-components/resolvers')

module.exports = {
    chainWebpack: config =>{
        config.plugin('html').tap(args => {
            args[0].title = 'AR互动游戏'
            return args
        })
    },
    configureWebpack: {
        plugins: [
            AutoImport({
                resolvers: [ElementPlusResolver()],
            }),
            Components({
                resolvers: [ElementPlusResolver()],
            }),
        ],
    },
    devServer: {
        port : 8081,
        disableHostCheck: true,
        proxy: {  //配置跨域
            '/api': {
                target: 'http://localhost:8080',
                changOrigin: true,  //允许跨域
                pathRewrite: {
                    '^/api': ''
                }
            }
        }
    },

}