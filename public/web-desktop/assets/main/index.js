System.register("chunks:///_virtual/PlayerController.ts",["./_rollupPluginModLoBabelHelpers.js","cc"],(function(e){"use strict";var t,s,i,o,r,n,p,c,a,u,h,l,_,y,m;return{setters:[function(e){t=e.applyDecoratedDescriptor,s=e.inheritsLoose,i=e.defineProperty,o=e.assertThisInitialized,r=e.initializerDefineProperty},function(e){n=e.cclegacy,p=e._decorator,c=e.Animation,a=e.Node,u=e.ParticleSystem,h=e.input,l=e.Input,_=e.KeyCode,y=e.Vec3,m=e.Component}],execute:function(){var d,f,P,v,C,E,S,K,g;n._RF.push({},"1ec6a/x1HVJVqPUbmi1k5OJ","PlayerController",void 0);var J=p.ccclass,T=p.property;e("PlayerController",(d=J("PlayerController"),f=T({type:c}),P=T({type:a}),v=T({type:u}),d((S=t((E=function(e){function t(){for(var t,s=arguments.length,n=new Array(s),p=0;p<s;p++)n[p]=arguments[p];return t=e.call.apply(e,[this].concat(n))||this,i(o(t),"_startJump",!1),i(o(t),"_startCharge",!1),i(o(t),"_IsCompressed",!1),i(o(t),"_curJumpSpeed",0),i(o(t),"_curPos",new y),i(o(t),"_deltaPos",new y(0,0,0)),i(o(t),"_targetPos",new y),i(o(t),"_acceleratedSpeed",25),i(o(t),"_curJumpTime",0),i(o(t),"_jumpTime",.5),i(o(t),"_curCompressLevel",1),i(o(t),"_compressSpeed",1),i(o(t),"_compressLimit",.4),i(o(t),"_releaseSpeed",4),r(o(t),"BodyAnim",S,o(t)),r(o(t),"LocalBody",K,o(t)),r(o(t),"particleSys",g,o(t)),t}s(t,e);var n=t.prototype;return n.start=function(){this.particleSys.enabled=!1,h.on(l.EventType.KEY_UP,this.onKeyUp,this),h.on(l.EventType.KEY_DOWN,this.onKeyDown,this)},n.setInputActive=function(e){e?(h.on(l.EventType.KEY_UP,this.onKeyUp,this),h.on(l.EventType.KEY_DOWN,this.onKeyDown,this)):(h.off(l.EventType.KEY_UP,this.onKeyUp,this),h.off(l.EventType.KEY_DOWN,this.onKeyDown,this))},n.onKeyUp=function(e){e.keyCode===_.SPACE&&this.jump()},n.onKeyDown=function(e){e.keyCode===_.SPACE&&this.charge()},n.jump=function(){this.particleSys.enabled=!1,this._startJump=!0,this._startCharge=!1,this._curJumpTime=0,this.node.getPosition(this._curPos);var e=this._curJumpSpeed*this._jumpTime;y.add(this._targetPos,this._curPos,new y(e,0,0)),h.off(l.EventType.KEY_UP,this.onKeyUp,this),h.off(l.EventType.KEY_DOWN,this.onKeyDown,this)},n.charge=function(){this.particleSys.enabled=!0,this.particleSys.clear(),h.on(l.EventType.KEY_UP,this.onKeyUp,this),this._startCharge=!0,this._IsCompressed=!0,this._curJumpSpeed=0},n.onOnceJumpEnd=function(){h.on(l.EventType.KEY_DOWN,this.onKeyDown,this),this.node.emit("JumpEnd",this._targetPos.x)},n.update=function(e){if(this._startCharge)this._curJumpSpeed+=this._acceleratedSpeed*e,this._curCompressLevel>this._compressLimit&&(this._curCompressLevel-=this._compressSpeed*e),this.LocalBody.setScale(new y(1,this._curCompressLevel,1));else if(this._IsCompressed){var t;if(this._curCompressLevel+=this._releaseSpeed*e,this._curCompressLevel>=1)this.LocalBody.setScale(new y(1,1,1)),this._IsCompressed=!1,null===(t=this.BodyAnim)||void 0===t||t.play("jump");else this.LocalBody.setScale(new y(1,this._curCompressLevel,1))}else this._startJump&&(this._curJumpTime+=e,this._curJumpTime>this._jumpTime?(this.node.setPosition(this._targetPos),this._startJump=!1,this.onOnceJumpEnd()):(this.node.getPosition(this._curPos),this._deltaPos.x=this._curJumpSpeed*e,y.add(this._curPos,this._curPos,this._deltaPos),this.node.setPosition(this._curPos)))},t}(m)).prototype,"BodyAnim",[f],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),K=t(E.prototype,"LocalBody",[P],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),g=t(E.prototype,"particleSys",[v],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),C=E))||C));n._RF.pop()}}}));

System.register("chunks:///_virtual/GameManager.ts",["./_rollupPluginModLoBabelHelpers.js","cc","./PlayerController.ts"],(function(e){"use strict";var t,r,i,n,o,a,l,u,s,h,c,d,p,b,y,f,C;return{setters:[function(e){t=e.applyDecoratedDescriptor,r=e.inheritsLoose,i=e.createClass,n=e.initializerDefineProperty,o=e.assertThisInitialized,a=e.defineProperty},function(e){l=e.cclegacy,u=e._decorator,s=e.Prefab,h=e.ParticleSystem,c=e.Animation,d=e.Node,p=e.Label,b=e.Vec3,y=e.instantiate,f=e.Component},function(e){C=e.PlayerController}],execute:function(){var g,B,m,L,P,R,S,v,x,A,G,M,N,_,E;l._RF.push({},"4dd827DVVtNFYC9903UukM/","GameManager",void 0);var w,I=u.ccclass,z=u.property;!function(e){e[e.GS_PLAYING=0]="GS_PLAYING",e[e.GS_END=1]="GS_END"}(w||(w={}));e("GameManager",(g=I("GameManager"),B=z({type:s}),m=z({type:C}),L=z({type:h}),P=z({type:c}),R=z({type:d}),S=z({type:p}),g((A=t((x=function(e){function t(){for(var t,r=arguments.length,i=new Array(r),l=0;l<r;l++)i[l]=arguments[l];return t=e.call.apply(e,[this].concat(i))||this,n(o(t),"cubePref",A,o(t)),n(o(t),"playerCtrl",G,o(t)),n(o(t),"playerParticle",M,o(t)),n(o(t),"BodyAnim",N,o(t)),n(o(t),"endMenu",_,o(t)),n(o(t),"scoreLabel",E,o(t)),a(o(t),"curCubeLeftBorder",0),a(o(t),"curCubeRightBorder",0),a(o(t),"nextCubeLeftBorder",0),a(o(t),"nextCubeRightBorder",0),a(o(t),"playerScore",0),t}r(t,e);var l=t.prototype;return l.start=function(){var e;this.curState=w.GS_PLAYING,null===(e=this.playerCtrl)||void 0===e||e.node.on("JumpEnd",this.onPlayerJumpEnd,this)},l.init=function(){this.endMenu&&(this.endMenu.active=!1),this.generateRoad(),this.playerCtrl&&(this.playerCtrl.setInputActive(!1),this.playerCtrl.node.setPosition(b.ZERO),this.BodyAnim.node.setPosition(new b(0,1,0)),this.BodyAnim.node.setRotationFromEuler(b.ZERO),this.playerScore=0,this.scoreLabel.string=""+this.playerScore)},l.generateRoad=function(){this.node.destroyAllChildren();var e=Math.floor(2*Math.random())+2,t=y(this.cubePref);this.node.addChild(t),t.setScale(e,1,1),t.setPosition(0,-1.5,0),this.generateNextCube(e/2),this.curCubeLeftBorder=-e/2,this.curCubeRightBorder=e/2},l.generateNextCube=function(e){var t=Math.floor(2*Math.random())+2,r=Math.floor(4*Math.random())+2;this.curCubeLeftBorder=this.nextCubeLeftBorder,this.curCubeRightBorder=this.nextCubeRightBorder,this.nextCubeLeftBorder=e+r,this.nextCubeRightBorder=e+r+t;var i=(this.nextCubeLeftBorder+this.nextCubeRightBorder)/2,n=y(this.cubePref);this.node.addChild(n),n.setPosition(i,-1.5,0),n.setScale(t,1,1),n.getChildByName("Body").getComponent(c).play()},l.onPlayerJumpEnd=function(e){var t=this;if(e>this.curCubeLeftBorder&&e<this.curCubeRightBorder);else if(e>this.nextCubeLeftBorder&&e<this.nextCubeRightBorder)this.generateNextCube(this.nextCubeRightBorder),this.playerScore+=1,this.scoreLabel.string=""+this.playerScore;else{var r=e>this.curCubeRightBorder&&e<this.curCubeRightBorder+.5,i=e>this.nextCubeRightBorder&&e<this.nextCubeRightBorder+.5,n=e<this.nextCubeLeftBorder&&e>this.nextCubeLeftBorder-.5;r||i?this.BodyAnim.play("fallRight"):n?this.BodyAnim.play("fallLeft"):this.BodyAnim.play("fall"),this.curState=w.GS_END,setTimeout((function(){t.curState=w.GS_PLAYING}),3e3)}},i(t,[{key:"curState",set:function(e){switch(e){case w.GS_PLAYING:this.init(),this.playerCtrl.setInputActive(!0),this.endMenu.active=!1;break;case w.GS_END:this.playerCtrl.setInputActive(!1),this.endMenu.active=!0}}}]),t}(f)).prototype,"cubePref",[B],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),G=t(x.prototype,"playerCtrl",[m],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),M=t(x.prototype,"playerParticle",[L],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),N=t(x.prototype,"BodyAnim",[P],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),_=t(x.prototype,"endMenu",[R],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),E=t(x.prototype,"scoreLabel",[S],{configurable:!0,enumerable:!0,writable:!0,initializer:function(){return null}}),v=x))||v));l._RF.pop()}}}));

System.register("chunks:///_virtual/main",["./PlayerController.ts","./GameManager.ts"],(function(){"use strict";return{setters:[null,null],execute:function(){}}}));

(function(r) {
  r('virtual:///prerequisite-imports/main', 'chunks:///_virtual/main'); 
})(function(mid, cid) {
    System.register(mid, [cid], function (_export, _context) {
    return {
        setters: [function(_m) {
            var _exportObj = {};

            for (var _key in _m) {
              if (_key !== "default" && _key !== "__esModule") _exportObj[_key] = _m[_key];
            }
      
            _export(_exportObj);
        }],
        execute: function () { }
    };
    });
});