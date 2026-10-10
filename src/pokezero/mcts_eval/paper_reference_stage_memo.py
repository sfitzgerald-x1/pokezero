"""Stage-local memo for deterministic inputs at identical particle owners.

Only resampling multiplicity is deduplicated. Different particles/stages never
share entries; all opponent actions and chance trials still draw independently.
"""
class StageIdentityMemo:
    def __init__(self, receipt):
        self.stage = None
        self.entries = {}
        self.receipt = receipt
        receipt.update(hits=0, misses=0, scope="identical resampled particle object within one stage")

    def get(self, particle, stage, build):
        if self.stage != stage:
            self.entries.clear()
            self.stage = stage
        key=id(particle)
        if key in self.entries:
            owner, value=self.entries[key]
            if owner is not particle:
                raise RuntimeError("historical input memo ownership drift")
            self.receipt["hits"]+=1
            return value
        value=build()
        self.entries[key]=(particle,value)
        self.receipt["misses"]+=1
        return value
