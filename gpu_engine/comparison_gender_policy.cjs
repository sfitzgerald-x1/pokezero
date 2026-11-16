// The comparator's prospective constructor law, using its already-bound Dex.
// No team generation or historical gender recovery happens here.
const fs=require('node:fs');
const {load}=require('./team_rows.cjs');
const {buildPolicy}=require('./gender_policy.cjs');
const metadata=JSON.parse(fs.readFileSync(0,'utf8'));
const pool=load(process.argv[2]);
if(JSON.stringify(pool.vocabulary)!==JSON.stringify(metadata.vocabulary))
  throw Error('comparison gender vocabulary differs');
process.stdout.write(JSON.stringify(buildPolicy(pool.dex,metadata.vocabulary.species))+'\n');
