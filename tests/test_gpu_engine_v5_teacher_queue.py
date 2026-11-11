"""Inert concurrency controls; no real roots, actors, games or accelerator."""
from threading import Event,get_ident
from types import SimpleNamespace as NS
import unittest

from gpu_engine.v5_teacher_queue import LabelQueue


class LabelQueueTests(unittest.TestCase):
    def test_collection_can_advance_and_publication_stays_parent_fifo(self):
        entered=Event();release=Event();parent=get_ident();seen=[];published=[];checks=[]
        roots=[object(),object()];params=object();bank=object()
        class Provider:
            def reset_metrics(self):self.current=[]
            def label_chunk(self,root,frame,b,p):
                self_thread=get_ident();seen.append((root,frame,b,p,self_thread))
                if frame==0:entered.set();release.wait(5)
                self.current.append(frame);return [frame]
            def metrics(self):return dict(frames=list(self.current))
        queue=LabelQueue(Provider(),capacity=2,check_progress=lambda:checks.append(True),
            publish=lambda host,labels,metrics,seconds:published.append((host.frames,labels,metrics,get_ident())))
        try:
            queue.submit(roots[0],NS(frames=0),bank,params)
            self.assertTrue(entered.wait(3))
            # This is the next acting chunk while the first labels are blocked.
            queue.submit(roots[1],NS(frames=1),bank,params)
            self.assertEqual(published,[]);self.assertEqual(queue.maximum_pending,2)
            release.set();queue.drain()
        finally:release.set();queue.close()
        self.assertEqual([(x[0],x[1],x[2])for x in published],[(0,[0],{'frames':[0]}),(1,[1],{'frames':[1]})])
        self.assertTrue(all(x[3]==parent for x in published))
        self.assertTrue(all(x[4]!=parent for x in seen));self.assertEqual(len({x[4]for x in seen}),1)
        for index,value in enumerate(seen):
            self.assertIs(value[0],roots[index]);self.assertIs(value[2],bank);self.assertIs(value[3],params)
        self.assertTrue(checks);self.assertEqual(queue.report()['chunks'],2)

    def test_capacity_backpressure_publishes_before_accepting_more(self):
        provider=NS(reset_metrics=lambda:None,label_chunk=lambda roots,frames,*a:[frames],metrics=lambda:{})
        published=[];queue=LabelQueue(provider,capacity=1,check_progress=lambda:None,publish=lambda host,*a:published.append(host.frames))
        try:
            queue.submit(object(),NS(frames=0),None,None)
            queue.submit(object(),NS(frames=1),None,None)
            self.assertEqual(published,[0]);self.assertEqual(len(queue.pending),1)
            queue.drain();self.assertEqual(published,[0,1]);self.assertEqual(queue.maximum_pending,1)
        finally:queue.close()
        with self.assertRaisesRegex(ValueError,'closed'):queue.submit(None,None,None,None)

    def test_producer_timeout_error_is_not_misread_as_wait_timeout(self):
        def fail(*a):raise TimeoutError('actual label producer failure')
        provider=NS(reset_metrics=lambda:None,label_chunk=fail,metrics=lambda:{})
        queue=LabelQueue(provider,capacity=2,check_progress=lambda:None,publish=lambda *a:self.fail('failed labels were published'))
        try:
            queue.submit(object(),NS(frames=0),None,None)
            with self.assertRaisesRegex(TimeoutError,'actual label producer'):queue.drain()
        finally:queue.close(failed=True)

    def test_wait_retains_completed_work_stall_guard(self):
        entered=Event();release=Event();checks=[]
        def label(*a):entered.set();release.wait(5);return []
        def guard():
            checks.append(True)
            if len(checks)>2:raise RuntimeError('original completed-work stall')
        provider=NS(reset_metrics=lambda:None,label_chunk=label,metrics=lambda:{})
        queue=LabelQueue(provider,capacity=1,check_progress=guard,publish=lambda *a:self.fail('stalled work was published'))
        try:
            queue.submit(object(),NS(frames=0),None,None);self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(RuntimeError,'completed-work stall'):queue.drain()
        finally:release.set();queue.close()
        for capacity in (-1,0,True,5):
            with self.assertRaises(ValueError):LabelQueue(provider,capacity=capacity,check_progress=guard,publish=lambda *a:None)


if __name__=='__main__':unittest.main()
