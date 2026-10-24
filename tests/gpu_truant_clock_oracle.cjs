// Pinned Gen3 authored clock boundaries, never a source/game-strength panel.
const path=require('node:path');
const {Battle}=require(path.join(process.argv[2],'dist/sim/battle.js'));
function create(kind='lead') {
 const b=new Battle({formatid:'gen3customgame',seed:[1,2,3,4]});
 const slak={species:'Slaking',ability:'Truant',moves:['slackoff','splash','hyperbeam'],level:100};
 const ditto={species:'Ditto',ability:'Limber',moves:['splash'],level:100};
 const lead=kind==='residual'?{species:'Shedinja',ability:'Wonder Guard',moves:['splash'],level:100}:ditto;
 b.setPlayer('p1',{name:'clock',team:kind==='lead'?[slak,ditto]:[lead,slak]});
 b.setPlayer('p2',{name:'control',team:[{species:kind==='residual'?'Tyranitar':'Ditto',
  ability:kind==='residual'?'Sand Stream':'Limber',moves:['splash','seismictoss'],level:100}]});
 for(const side of b.sides)for(const p of side.pokemon){
  p.baseMaxhp=p.maxhp=p.hp=1000;
  for(const key of Object.keys(p.storedStats))p.storedStats[key]=200;
  p.storedStats.spe=side.id==='p1'?301:199;p.updateSpeed();
  for(const slot of p.moveSlots)slot.pp=8;
 }
 const holder=b.p1.pokemon.find(p=>p.species.id==='slaking');holder.hp=100;
 if(kind==='midturn'||kind==='residual')b.p1.active[0].hp=1;
 if(kind==='midturn'){b.p2.active[0].storedStats.spe=501;b.p2.active[0].updateSpeed();}
 // Accuracy succeeds, crits fail; only the explicitly requested paralysis case
 // below changes this. No naturally sampled chance or learned policy is used.
 b.randomChance=(n,d)=>n>=d;b.randomizer=x=>x;
 return b;
}
function snapshot(b,start=0) {
 const p=b.p1.active[0];
 return {turn:b.turn,request:b.requestState,species:p.species.id,hp:p.hp,
  phase:p.truantTurn??null,status:p.status,recharge:!!p.volatiles.mustrecharge,
  pp:p.moveSlots.map(s=>s.pp),events:b.log.slice(start).filter(s=>/^\|(move|cant|switch|faint|upkeep|turn)\|?/.test(s))};
}
function ordinary(mode) {
 const b=create(),p=b.p1.active[0];
 if(mode==='sleep'){p.setStatus('slp');p.statusState.time=5;}
 if(mode==='paralysis'){p.setStatus('par');b.randomChance=(n,d)=>(n===1&&d===4)||n>=d;}
 if(mode==='recharge') {
  p.addVolatile('mustrecharge'); // authored latent boundary, refresh its request
  b.makeRequest('move');
 }
 const rows=[snapshot(b)];
 for(let i=0;i<4;i++) {const start=b.log.length;
  b.makeChoices(p.volatiles.mustrecharge?'move recharge':'move slackoff','move splash');
  rows.push(snapshot(b,start));}
 b.destroy();return rows;
}
function switching(kind) {
 const b=create(kind),rows=[snapshot(b)];
 function choose(a,c){const start=b.log.length;b.makeChoices(a,c);rows.push(snapshot(b,start));}
 if(kind==='ordinary')choose('switch 2','move splash');
 else {choose('move splash',kind==='midturn'?'move seismictoss':'move splash');choose('switch 2',undefined);}
 choose('move slackoff','move splash');choose('move slackoff','move splash');
 b.destroy();return rows;
}
process.stdout.write(JSON.stringify({schema:'pokezero.showdown-gen3-truant-clock-fixtures.v1',
 scope:'Seven finite authored boundary traces; recharge is an injected latent boundary, not Hyper Beam coverage; no Trace/Skill Swap qualification.',
 startup_and_residual:ordinary('normal'),sleep:ordinary('sleep'),paralysis:ordinary('paralysis'),
 recharge:ordinary('recharge'),ordinary_switch:switching('ordinary'),
 midturn_forced:switching('midturn'),postresidual_forced:switching('residual')}));
