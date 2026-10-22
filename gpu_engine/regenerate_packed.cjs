// Reconstruct original real sets, including fields omitted from the tensor pool.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {Teams}=require(path.join(process.argv[2],'dist/sim/teams.js'));
const seeds=JSON.parse(fs.readFileSync(0,'utf8'));
if(!Array.isArray(seeds)||!seeds.length||seeds.length>400||new Set(seeds).size!==seeds.length
  ||seeds.some(x=>!Number.isSafeInteger(x)||x<0||x>=2**32))throw Error('bounded unique original seeds required');
process.stdout.write(JSON.stringify(seeds.map(seed=>{
  const bytes=crypto.createHash('sha256').update('pokezero-gpu-pool-v1:'+seed).digest();
  const rng=Array.from({length:4},(_,k)=>bytes.readUInt16LE(2*k));
  return {seed,packed:Teams.pack(Teams.generate('gen3randombattle',{seed:rng}))};
})));
