import json
from pathlib import Path
from .inference.gateway import InferenceGateway
from .plugins import load_plugin
from .run_store import create_run,finish_run,file_hash
from .video import read_video,VideoWriter


def run(config,logic_snapshot=None):
    context=create_run(config,logic_snapshot)
    gateway=None
    writer=None
    count=0
    try:
        for name in ['events.jsonl','trace.jsonl']:
            (context.path/name).touch()
        video_hash=file_hash(config.input.path)
        gateway=InferenceGateway(config,context.path,video_hash)
        plugin=load_plugin(config,gateway,context.path,context.path/'logic')
        for frame in read_video(config.input.path,config.input.start_ms,config.input.end_ms):
            plugin.tick(frame)
            if config.output.video:
                if writer is None:
                    h,w=frame.pixels.shape[:2]
                    writer=VideoWriter(context.path/'output.mp4',w,h)
                writer.write(plugin.render(frame),frame.time)
            count+=1
        if count==0:
            raise ValueError('Input segment contains no decoded frames')
        if writer:
            active_writer=writer
            writer=None
            active_writer.close()
        metrics={**plugin.finalize(),'frames':count,'providers':gateway.counts,'backends':gateway.backends}
        (context.path/'metrics.json').write_text(json.dumps(metrics,indent=2))
        model_manifests=gateway.registry
        active_gateway=gateway
        gateway=None
        active_gateway.close()
        finish_run(context,'completed',{'input_hash':video_hash,'logic_source_hash':file_hash(
            context.path/'logic'/'logic.py'),
            'metrics':metrics,'model_manifests':model_manifests})
        return context
    except BaseException as error:
        finish_run(context,'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                   {'error_type':type(error).__name__,'error':str(error)[:500],'frames':count})
        raise
    finally:
        for resource in [writer,gateway]:
            if resource:
                try:
                    resource.close()
                except Exception:
                    # Primary exception/status remains authoritative; try every resource.
                    pass
