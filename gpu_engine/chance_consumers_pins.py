"""chance_consumers' reviewed pins: whole-file sha256 of the sources its consumer proof reads (SOURCE_PINS) and of each
transition-domain module, in chance_consumers.DOMAIN_MODULES order (DOMAIN_PINS). They live apart from chance_consumers.py
so that ordinary_law's whole-file pin of chance_consumers.py covers its logic, not every refresh of these values (one
domain edit used to re-pin chance_consumers.py, and through it ordinary_law, in every branch at once). chance_consumers
checks both tables against the files' bytes before every plan; nothing else reads them. Values are written by
scripts/refresh_source_pins.py, one entry per line; which files are pinned is a review decision made by hand."""
SOURCE_PINS = {
    'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407',
    'critical.py': 'b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e',
    'effects.py': '47f549d608b60d46e8cc4db7bad5be249ae728a4aab2a0ef28f105f3614e40e2',
    'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd',
    'entry.py': 'eabf2713a100256ba86fd0b5ac4155042e2a0c6dfe0bc37c7f0c6219dbdd22c1',
    'mechanics.py': 'e8476cc3e0029f3ed0bf82276312440e6dd8b340ffeb0c68cdd5611754eef661',
    'phazing.py': 'c27039f267bd18660adb3c2bd826e58a7da953eea0e79677de5428f042447969',
    'pursuit.py': '2b01102bb3819791fc1edea0ba0fe0a0b913e8dfa5a8f6501f48f3fe260a8309',
    'residual_order.py': '7b1a958d4ce27cf8fe7560d8f5a4f95d5cb54a17adbe23ba72dccc00e35bf454',
    'secondary.py': 'ada20a946cc106dcf798fa053392ed895fe778c977af13774ab58e0c8c657fe6',
    'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154',
    'state.py': '67ca721ce1f6b1a5c3e9c6fd42fe33d65f62ffcec006f5c5efd188225111a59c',
    'transition.py': '820fdcc40285b362cbfacb5c91749e7b20446543ac63278afc160d0bd80f143f',
    'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5',
}
DOMAIN_PINS = {
    'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407',
    'absorption.py': '2d684602eb9aa69bc470b215c188533b0e1c4d8495dd5b71ca2508dc38f20623',
    'baton.py': 'f86ad12ae0d59c1c8f89eb3f84f2d0519e1214ae4a4506d5cb700c6e0c20fa01',
    'combat.py': '3e802d3fd2f488e9aa2a8ead663f935158c5fec1b7fb745034e54f21d38015af',
    'critical.py': 'b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e',
    'damage_moves.py': 'c2427b88dc7f7b643625659f12498e1c06412c6875c60d548fa573f6e7a45e72',
    'effects.py': '47f549d608b60d46e8cc4db7bad5be249ae728a4aab2a0ef28f105f3614e40e2',
    'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd',
    'entry.py': 'eabf2713a100256ba86fd0b5ac4155042e2a0c6dfe0bc37c7f0c6219dbdd22c1',
    'focus_punch.py': '881bd17a84dc22576aa618256ada2b0f67bee29224dbb50508925b55e119585f',
    'items.py': 'e6813b7460655965e194a058c4f279ef5cb684281c30f918ef5bc89beb030bfe',
    'linked_traps.py': '743c01ee69509162e7829ae555b85f069f1c20ec9bd19cec0779d066894bade4',
    'mechanics.py': 'e8476cc3e0029f3ed0bf82276312440e6dd8b340ffeb0c68cdd5611754eef661',
    'phazing.py': 'c27039f267bd18660adb3c2bd826e58a7da953eea0e79677de5428f042447969',
    'pressure.py': '73eefb16678fedf1885a4e11cbc8954d15706592742caa6b0b673fb4d63d698c',
    'pursuit.py': '2b01102bb3819791fc1edea0ba0fe0a0b913e8dfa5a8f6501f48f3fe260a8309',
    'residual_order.py': '7b1a958d4ce27cf8fe7560d8f5a4f95d5cb54a17adbe23ba72dccc00e35bf454',
    'secondary.py': 'ada20a946cc106dcf798fa053392ed895fe778c977af13774ab58e0c8c657fe6',
    'self_destruct.py': 'b3c9df0ea093eca7c0960b4c4ffd4277e289a0ded6f32f914466c0752fa9b739',
    'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154',
    'solar_beam.py': '5463e103392e24df7cc560003740b7e425aba766a0d1bba90ff4428606849155',
    'state.py': '67ca721ce1f6b1a5c3e9c6fd42fe33d65f62ffcec006f5c5efd188225111a59c',
    'sustain.py': '0b575a55f091312679058e0ab990116b1f2348f163654ae777e293c1e7b51b24',
    'transition.py': '820fdcc40285b362cbfacb5c91749e7b20446543ac63278afc160d0bd80f143f',
    'variable_moves.py': '0d3655aac5b547782457be46ebfee0b8b66ece3a360d05d4ec34cebbf801ea6d',
    'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5',
}
