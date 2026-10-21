import { Chess } from './vendor/chess.js';
const game = new Chess();
const el = id => document.getElementById(id);
const glyphs = {"w": {"k": "\u2654", "q": "\u2655", "r": "\u2656", "b": "\u2657", "n": "\u2658", "p": "\u2659"}, "b": {"k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f"}};
let history=[], selected=null, targets=[], flipped=false, busy=false, connected=false, generation=0, last=[];
function message(text) { el('status').textContent=text; }
function render() {
 const files=flipped?'hgfedcba':'abcdefgh', ranks=flipped?'12345678':'87654321';
 el('board').replaceChildren();
 for(const [row,rank] of [...ranks].entries()) for(const [col,file] of [...files].entries()) {
  const square=file+rank, piece=game.get(square), button=document.createElement('button');
  button.className='square'+((file.charCodeAt(0)+Number(rank))%2===0?' dark':'')+(square===selected?' selected':'')+(last.includes(square)?' last':'');
  if(targets.some(m=>m.to===square)) button.className+=' legal'+(piece?' capture':'');
  button.setAttribute('aria-label',square+(piece?' '+(piece.color==='w'?'white':'black')+' '+({k:'king',q:'queen',r:'rook',b:'bishop',n:'knight',p:'pawn'}[piece.type]):' empty'));
  if(piece){const span=document.createElement('span');span.className='piece '+(piece.color==='w'?'white':'black');span.textContent=glyphs[piece.color][piece.type];button.append(span);}
  if(row===7){const span=document.createElement('span');span.className='coord';span.textContent=file;button.append(span);}
  if(col===0){const span=document.createElement('span');span.className='coord rank';span.textContent=rank;button.append(span);}
  button.addEventListener('click',()=>click(square));el('board').append(button);
 }
 el('moves').replaceChildren();const sans=game.history();
 for(let i=0;i<sans.length;i+=2){const li=document.createElement('li');li.textContent=sans[i]+' '+(sans[i+1]||'');el('moves').append(li);}
 el('moves').scrollTop=el('moves').scrollHeight;
 el('undo').disabled=busy||!history.length||(el('side').value==='b'&&history.length===1);el('model').disabled=busy;el('side').disabled=busy;
}
function status() {
 if(game.isCheckmate())message('Checkmate. '+(game.turn()==='w'?'Black':'White')+' wins.');
 else if(game.isStalemate())message('Draw by stalemate.');
 else if(game.isDraw())message('Game drawn.');
 else message((game.turn()===el('side').value?'Your turn':'Model\u2019s turn')+(game.isCheck()?' ? check.':'.'));
}
function promotion(color) {
 return new Promise(resolve=>{
  const dialog=el('promotion');el('promotion-options').replaceChildren();
  for(const type of ['q','r','b','n']){const b=document.createElement('button');b.textContent=glyphs[color][type];b.setAttribute('aria-label',({q:'Queen',r:'Rook',b:'Bishop',n:'Knight'})[type]);b.onclick=()=>{resolve(type);dialog.close();};el('promotion-options').append(b);}
  dialog.onclose=()=>resolve(null);
  dialog.oncancel=e=>{e.preventDefault();dialog.close();resolve(null);};dialog.showModal();
 });
}
async function click(square) {
 if(busy||!connected||game.isGameOver()||game.turn()!==el('side').value)return;
 const options=targets.filter(m=>m.to===square);
 if(selected&&options.length){
  const version=generation;let promote; if(options.some(m=>m.promotion)){promote=await promotion(game.turn());if(!promote||version!==generation)return;}
  const move=game.move({from:selected,to:square,...(promote?{promotion:promote}:{})});
  history.push(move.from+move.to+(move.promotion||''));last=[move.from,move.to];selected=null;targets=[];render();status();await modelMove();return;
 }
 const piece=game.get(square);selected=piece&&piece.color===game.turn()?square:null;
 targets=selected?game.moves({square:selected,verbose:true}):[];render();
}
async function modelMove() {
 if(game.isGameOver()||game.turn()===el('side').value||!connected)return;
 const version=generation;busy=true;el('retry').hidden=true;message('Model is thinking...');render();
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),60000);
 try {
  const response=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:el('model').value,moves:history}),signal:controller.signal});
  const data=await response.json();if(version!==generation)return;if(!response.ok)throw new Error(data.error||'Model unavailable.');
  const uci=data.move;const move=game.move({from:uci.slice(0,2),to:uci.slice(2,4),...(uci.length===5?{promotion:uci[4]}:{})});
  history.push(uci);last=[move.from,move.to];status();
 }catch(error){if(version===generation){message(error.name==='AbortError'?'The model took too long. Retry the move.':error.message);el('retry').hidden=false;}}
 finally{clearTimeout(timer);if(version===generation){busy=false;render();}}
}
function newGame(){generation++;if(el('promotion').open)el('promotion').close();busy=false;game.reset();history=[];selected=null;targets=[];last=[];flipped=el('side').value==='b';el('retry').hidden=true;render();status();modelMove();}
el('new').onclick=newGame;el('side').onchange=newGame;el('model').onchange=newGame;
el('flip').onclick=()=>{flipped=!flipped;render();};el('retry').onclick=modelMove;
el('undo').onclick=()=>{if(busy)return;game.undo();history.pop();if(history.length&&game.turn()!==el('side').value){game.undo();history.pop();}last=[];selected=null;targets=[];el('retry').hidden=true;render();status();};
async function models(){try{const response=await fetch('/api/models');if(!response.ok)throw new Error('Server unavailable');const data=await response.json();const chosen=el('model').value;el('model').replaceChildren();for(const m of data.models){const o=document.createElement('option');o.value=m.id;o.textContent=m.name;el('model').append(o);}if(data.models.some(m=>m.id===chosen))el('model').value=chosen;connected=data.models.length>0;if(connected&&!busy)status();else if(!connected)message('Waiting for the model to become available.');}catch{connected=false;if(!busy)message('Could not connect. Refresh to retry.');}}
render();await models();setInterval(models,30000);
