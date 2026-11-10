"""PyCharm Run: complete RGB/glass/door/box continuous navigation demo."""
import runtime
import argparse,json,sys
from pathlib import Path
from datetime import datetime
from dataclasses import replace
from configuration import Config,ROOT
from mission import Mission


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--headless',action='store_true');parser.add_argument('--speed',type=float,default=2.)
    parser.add_argument('--close-on-done',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--rate',type=float,help='public contact-work task cost, m/J')
    parser.add_argument('--scene',type=Path,help='evaluator scene JSON; only the physical builder consumes hidden parameters')
    parser.add_argument('--decision-mode',choices=('online','after-probe'),default='online',help='online: paired window updates and immediate action switches')
    args=parser.parse_args()
    if args.speed<=0:parser.error('speed must be positive')
    folder=args.output or ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    ui=None
    if not args.headless:
        from ui.live import LiveUI
        ui=LiveUI(args.speed)
    mission=Mission(replace(Config(),online_decisions=args.decision_mode=='online'),folder,ui,args.rate,args.scene)
    mission.write_json(folder/'launch.json',dict(interpreter=sys.executable,argv=sys.argv,cwd=str(Path.cwd()),root=str(ROOT)))
    try:
        okay=mission.run()
        from replay_report import build
        report=build(mission.folder)
        if okay and report.resolve()!=(ROOT/'report.html').resolve():
            import shutil
            shutil.copyfile(report,ROOT/'report.html')
        if ui and not args.close_on_done:ui.keep_open()
        return 0 if okay else 1
    finally:
        mission.close()
        if ui:ui.close()

if __name__=='__main__':raise SystemExit(main())
