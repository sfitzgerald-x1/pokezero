"""Literal occupancy and blocked runtime scopes for the finite device collector.

No repeated game, warmup trajectory, replacement cell or outcome-selected
sample is created. Corrected rate is a lane-occupancy normalization, not a new
full-game result. Host phase walls can overlap and are not summed into a total.
"""
import math
import numpy as np


class Counts:
    def __init__(self,batch_rows):
        if type(batch_rows)is not int or batch_rows<=0:raise ValueError('actual physical batch rows required')
        self.batch_rows=batch_rows;self.physical_steps=0;self.occupied_steps=0;self.requested_views=0
        self.game_requests=np.zeros(batch_rows,np.int64);self.seat_requests=np.zeros(batch_rows,np.int64)

    def add(self,requested):
        a=np.asarray(requested)
        if a.dtype!=np.bool_ or a.ndim!=3 or a.shape[1:]!=(self.batch_rows,2):
            raise ValueError('literal requested[step,batch,owner] bool array required')
        occupied=np.any(a,axis=2)
        self.physical_steps+=a.shape[0]*self.batch_rows;self.occupied_steps+=int(occupied.sum());self.requested_views+=int(a.sum())
        self.game_requests+=occupied.sum(axis=0);self.seat_requests+=a.sum(axis=(0,2))

    def report(self,status,*,kernel_seconds,collection_seconds):
        status=np.asarray(status)
        if status.shape!=(self.batch_rows,)or status.dtype.kind not in 'iu'or np.any((status<0)|(status>3)):
            raise ValueError('literal final game statuses required')
        for value in (kernel_seconds,collection_seconds):
            if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<0:
                raise ValueError('finite measured seconds required')
        complete=status==1;n=int(complete.sum());occupied=self.occupied_steps;physical=self.physical_steps
        describe=lambda a:None if not n else dict(mean=float(np.mean(a[complete])),p99=float(np.percentile(a[complete],99,method='linear')))
        raw=None if kernel_seconds<=0 else n/kernel_seconds
        return dict(scope='ACTUAL_DECLARED_COHORT_FULL_TAIL_NO_REFILL',batch_rows=self.batch_rows,
            complete=n,capped=int((status==2).sum()),refused=int((status==3).sum()),in_progress=int((status==0).sum()),
            physical_game_lane_steps=physical,occupied_game_lane_steps=occupied,requested_seat_views=self.requested_views,
            occupancy_fraction=None if not physical else occupied/physical,
            kernel_blocked_seconds=kernel_seconds,collection_wall_seconds=collection_seconds,
            completed_per_kernel_second=raw,
            lane_occupancy_corrected_completed_per_kernel_second=None if raw is None or not occupied else raw*physical/occupied,
            completed_per_collection_wall_second=None if collection_seconds<=0 else n/collection_seconds,
            completed_game_request_steps=describe(self.game_requests),completed_requested_seat_views=describe(self.seat_requests),
            p99_definition='linear percentile on complete games only; capped/refused remain in cohort denominators',
            corrected_rate_scope='normalizes the same full-tail kernel wall by physical/occupied game-lane steps; not a measured refilled throughput')


def aggregate(reports,game_end):
    """Count all actual banks; never average their rates or p99 values."""
    reports=tuple(reports)
    if not reports:return dict(status='UNMEASURED_NO_RUNTIME_REPORTS')
    integer=('complete','capped','refused','in_progress','physical_game_lane_steps','occupied_game_lane_steps','requested_seat_views')
    total={key:sum(r[key]for r in reports)for key in integer}
    seconds=sum(r['kernel_blocked_seconds']for r in reports)
    complete=total['complete'];physical=total['physical_game_lane_steps'];occupied=total['occupied_game_lane_steps']
    events=[x for x in game_end if x['status']=='COMPLETE']
    requests=[x['requests']for x in events if type(x.get('requests'))is int and x['requests']>0]
    summary=None if len(requests)!=complete or not complete else dict(mean=float(np.mean(requests)),p99=float(np.percentile(requests,99,method='linear')))
    return dict(status='MEASURED_ACTUAL_DECLARED_BANKS',**total,banks=len(reports),physical_batch_rows=[r['batch_rows']for r in reports],
        kernel_blocked_seconds=seconds,completed_per_kernel_second=None if seconds<=0 else complete/seconds,
        lane_occupancy_corrected_completed_per_kernel_second=None if seconds<=0 or not occupied else complete/seconds*physical/occupied,
        completed_engine_request_steps=summary,
        request_scope='original GAME_END requests for every COMPLETE game; no caps/refusals in mean/p99 but retained in full cohort',
        occupancy_scope='sum physical and requested.any(owner) occupied game-lane steps across declared banks, including final smaller bank',
        by_batch_rows={str(b):dict(banks=sum(r['batch_rows']==b for r in reports),complete=sum(r['complete']for r in reports if r['batch_rows']==b),
            kernel_blocked_seconds=sum(r['kernel_blocked_seconds']for r in reports if r['batch_rows']==b))for b in sorted(set(r['batch_rows']for r in reports))})
