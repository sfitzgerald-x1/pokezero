"""chance_consumers' reviewed pins: whole-file sha256 of the sources its consumer proof reads (SOURCE_PINS) and of each
transition-domain module, in chance_consumers.DOMAIN_MODULES order (DOMAIN_PINS). They live apart from chance_consumers.py
so that ordinary_law's whole-file pin of chance_consumers.py covers its logic, not every refresh of these values (one
domain edit used to re-pin chance_consumers.py, and through it ordinary_law, in every branch at once). chance_consumers
checks both tables against the files' bytes before every plan; nothing else reads them. Values are written by
scripts/refresh_source_pins.py, one entry per line; which files are pinned is a review decision made by hand."""
SOURCE_PINS = {
    'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407',
    'critical.py': '7f8be9394e5a5473df5e6e9e9dcd242947ac0970cf5f46d50e27b27658e58f72',
    'effects.py': '47f549d608b60d46e8cc4db7bad5be249ae728a4aab2a0ef28f105f3614e40e2',
    'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd',
    'entry.py': '688cc4eaa0266f71faf22c78fa2d91f2ee0fd8b875171e88fe4c2304c66aa285',
    'mechanics.py': 'fa63643594e8e01cae573c324fbd442fd82ddf63e4514bc23c2e3a4c337e3444',
    'phazing.py': '8004d4880991a074480b6a7e9be6f6cb980577a61d2388c4c8c56d9a1ab2f26a',
    'pursuit.py': '31a6805d9a6b0032c849877efcb2b5a2447b9abc29fbe5170e687b0abe1fbb55',
    'residual_order.py': 'd49fbd0f47ffe4fa1c98bba49eeaacfb5e16518c2ef938aa7ff311d996a0df94',
    'secondary.py': 'a3058e796b8affc38f5aa0d188bf52cc554d6c07cc8b4e83ac792ed70734001a',
    'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154',
    'state.py': '4cb9d910e8eb8cd70e80185fe4352132fd8933321a9fb00132623077b5993387',
    'transition.py': 'e86226e51f4ec2941d8ce793c6eaadae6deafaf591e2bd057eac1f03ce0e0a97',
    'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5',
}
DOMAIN_PINS = {
    'absorption.py': '2d684602eb9aa69bc470b215c188533b0e1c4d8495dd5b71ca2508dc38f20623',
    'action_history.py': 'a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407',
    'baton.py': 'f86ad12ae0d59c1c8f89eb3f84f2d0519e1214ae4a4506d5cb700c6e0c20fa01',
    'combat.py': '2d3ad693aebf879dc4b26cd992e9357fcc0e7a9fcf22bf1274e33869cd684aa7',
    'critical.py': '7f8be9394e5a5473df5e6e9e9dcd242947ac0970cf5f46d50e27b27658e58f72',
    'damage_moves.py': 'c2427b88dc7f7b643625659f12498e1c06412c6875c60d548fa573f6e7a45e72',
    'effects.py': '47f549d608b60d46e8cc4db7bad5be249ae728a4aab2a0ef28f105f3614e40e2',
    'encore.py': '63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd',
    'entry.py': '688cc4eaa0266f71faf22c78fa2d91f2ee0fd8b875171e88fe4c2304c66aa285',
    'focus_punch.py': '881bd17a84dc22576aa618256ada2b0f67bee29224dbb50508925b55e119585f',
    'forecast.py': '3c08324c4012ab60b8aa33f1a3d217c820783c4b85e0e0f5b0bdc2feade97bcf',
    'infatuation.py': '6d25d2cfe1f45e0cee4334e89b66ae2ff3ca92de141b2825d9345aec04bc3ffa',
    'items.py': '4a936d5d4e92036155bce8a1ae19005ca7555996cb23a6359cdd4c128fa93490',
    'linked_traps.py': '743c01ee69509162e7829ae555b85f069f1c20ec9bd19cec0779d066894bade4',
    'mechanics.py': 'fa63643594e8e01cae573c324fbd442fd82ddf63e4514bc23c2e3a4c337e3444',
    'phazing.py': '8004d4880991a074480b6a7e9be6f6cb980577a61d2388c4c8c56d9a1ab2f26a',
    'pressure.py': '73eefb16678fedf1885a4e11cbc8954d15706592742caa6b0b673fb4d63d698c',
    'pursuit.py': '31a6805d9a6b0032c849877efcb2b5a2447b9abc29fbe5170e687b0abe1fbb55',
    'residual_order.py': 'd49fbd0f47ffe4fa1c98bba49eeaacfb5e16518c2ef938aa7ff311d996a0df94',
    'secondary.py': 'a3058e796b8affc38f5aa0d188bf52cc554d6c07cc8b4e83ac792ed70734001a',
    'self_destruct.py': 'b3c9df0ea093eca7c0960b4c4ffd4277e289a0ded6f32f914466c0752fa9b739',
    'sleep_talk.py': '968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154',
    'solar_beam.py': '5463e103392e24df7cc560003740b7e425aba766a0d1bba90ff4428606849155',
    'state.py': '4cb9d910e8eb8cd70e80185fe4352132fd8933321a9fb00132623077b5993387',
    'sustain.py': '0b575a55f091312679058e0ab990116b1f2348f163654ae777e293c1e7b51b24',
    'transform.py': '5ff5f4613ec84b6ca7a72efec3ffd6b64dcf28d1f1672c633428a8e5dc45e20b',
    'transition.py': 'e86226e51f4ec2941d8ce793c6eaadae6deafaf591e2bd057eac1f03ce0e0a97',
    'variable_moves.py': '0d3655aac5b547782457be46ebfee0b8b66ece3a360d05d4ec34cebbf801ea6d',
    'weather.py': '3c861027e16d27917fa47fa5cf5d6dc0c68d3a74af282108dfb94369d4db57e5',
}
