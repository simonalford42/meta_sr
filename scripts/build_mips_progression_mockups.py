#!/usr/bin/env python3
"""Four MIPS progression figures using the archived ten-seed evaluations."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mips_progression_mockups'
METHODS = {
    'PySR': ('runs/709714/final_eval_baseline_10seed/eval_summary.json', 'runs/709714/final_eval_baseline_10seed/slurm_pysr/eval_0000'),
    'SRBench bundle': ('runs/709715/final_eval_mips_native_10seed/eval_summary.json', 'runs/709715/final_eval_mips_native_10seed/slurm_pysr/eval_0000'),
    'MIPS meta-evolution': ('runs/709714/final_eval_summary.json', 'runs/709714/final_eval/slurm_pysr/eval_0000'),
}
INK='#172d3a'; GRAY='#8c9ba5'; PALE='#e7edf0'; BLUE='#2784b0'; ORANGE='#e6a13b'; GREEN='#208976'; RED='#bb7980'


def data():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance=[]
    def read(p):
        b=(ROOT/p).read_bytes(); provenance.append({'path':p,'sha256':hashlib.sha256(b).hexdigest()}); return json.loads(b)
    original=read('outputs/mips_reproduction_all/summary.json')
    solved0={t['task'] for t in original['tasks'] if t['independent_success']}
    split=(ROOT/'splits/mips_sr_targets_plus_refined.txt').read_text().splitlines()
    groups=defaultdict(list)
    for c in split: groups[c.split(':')[1]].append(c)
    eligible=set(groups)-solved0
    results={}; rows=[]
    for method,(summary_path,raw_path) in METHODS.items():
        summary=read(summary_path); tasks=read(raw_path+'/tasks.json'); indexed={}
        assert len(tasks)==510 and summary['n_runs']==10
        for i,t in enumerate(tasks):
            path=raw_path+f'/results/task_{i:06d}.json'; r=read(path)
            key=(r['dataset_name'],r['run_index']); assert key==(t['dataset_name'],t['run_index']) and key not in indexed
            assert t['pysr_kwargs']['max_evals']==1000000 and t['pysr_kwargs']['timeout_in_seconds']==500
            indexed[key]=r.get('gt_match_score')==1
            rows.append(dict(method=method,dataset=key[0],run_index=key[1],seed=t['seed'],exact=int(indexed[key]),error=r.get('error'),source=path))
        assert set(indexed)=={(c,s) for c in split for s in range(10)}
        group_seeds={g:[s for s in range(10) if all(indexed[c,s] for c in cs)] for g,cs in groups.items()}
        solved={g for g,s in group_seeds.items() if s}&eligible
        details=next(v['result_details'] for v in summary.values() if isinstance(v,dict) and 'result_details' in v)
        assert sum(indexed.values())==sum(sum(s==1 for s in r['run_gt_scores']) for r in details)
        results[method]={'new_solved':sorted(solved),'group_exact_seeds':group_seeds,
            'scalar_solved_any_seed':sum(any(indexed[c,s] for s in range(10)) for c in split),
            'scalar_exact_fits':sum(indexed.values()),'scalar_total_fits':510}
    a,b,c=[set(r['new_solved']) for r in results.values()]
    assert len(solved0)==30 and len(eligible)==14 and [len(a),len(b),len(c)]==[5,9,11] and a<=b<=c
    partition={'original_solved':sorted(solved0),'outside_sr_candidate_set':sorted({t['task'] for t in original['tasks']}-solved0-eligible),
               'base_solved':sorted(a),'added_by_srbench':sorted(b-a),'added_by_mips':sorted(c-b),'remaining':sorted(eligible-c)}
    assert [len(v) for v in partition.values()]==[30,18,5,4,2,3]
    (OUT/'results.json').write_text(json.dumps({'partition':partition,'methods':results},indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (OUT/'all_1530_scalar_fits.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'all_62_problem_categories.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['task','category'])
        for k,ts in partition.items():
            for t in ts:w.writerow([t,k])
    return results


def setup(title, context=True):
    fig=plt.figure(figsize=(12,7),facecolor='white')
    fig.text(.065,.925,title,fontsize=23,fontweight='bold',color=INK)
    if context:
        fig.text(.065,.85,'62 problems',fontsize=13,fontweight='bold',color=INK)
        fig.text(.225,.85,'30 originally solved',fontsize=12,color=GRAY)
        fig.text(.445,.85,'32 unsolved → 18 excluded from SR + 14 candidates',fontsize=12,color=INK)
    return fig


def finish(fig,name):
    fig.text(.065,.055,'Solved = all components exact in at least one of 10 seeds. Sets are verified to be nested.',fontsize=9,color='#5b6972')
    fig.text(.065,.032,'1M evaluations / 500 s per fit. New solves use transition-table checks; full-program validation is separate.',fontsize=9,color='#5b6972')
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)


def staircase():
    fig=setup('From 5 to 11 solved: the effect of meta-evolution')
    ax=fig.add_axes([.09,.22,.82,.53]);x=[0,1,2];y=[5,9,11]
    ax.axhline(14,color=GRAY,ls='--',lw=1);ax.text(2.2,14.1,'14 candidates',ha='right',color=GRAY)
    ax.fill_between([0,1,2,2.22],[5,9,11,11],step='post',color=GREEN,alpha=.08)
    ax.step([0,1,2,2.22],[5,9,11,11],where='post',color=INK,lw=2.5)
    for xx,yy,col in zip(x,y,[BLUE,ORANGE,GREEN]):
        ax.scatter(xx,yy,s=180,c=col,zorder=5);ax.text(xx,yy+.6,str(yy),fontsize=28,fontweight='bold',color=col,ha='center')
    ax.text(.5,6.5,'+4 problems',color=ORANGE,fontsize=14,fontweight='bold',ha='center')
    ax.text(1.5,9.7,'+2 problems',color=GREEN,fontsize=14,fontweight='bold',ha='center')
    ax.annotate('',xy=(2.2,14),xytext=(2.2,11),arrowprops=dict(arrowstyle='|-|',color=RED))
    ax.text(2.28,12.5,'3 remain\nunsolved',color=RED,va='center',fontsize=11)
    ax.set_xticks(x,['Base PySR','SRBench bundle\n(PySR++)','MIPS meta-evolution'])
    ax.set_yticks([0,5,9,11,14]);ax.set_ylim(0,15.5);ax.set_xlim(-.25,2.65)
    ax.set_ylabel('Previously unsolved problems recovered');ax.grid(axis='y',alpha=.1)
    finish(fig,'01_staircase')


def waterfall():
    fig=setup('Meta-evolution adds six solves beyond base PySR')
    ax=fig.add_axes([.09,.22,.84,.54]);pos=np.arange(5);bottom=[0,5,9,0,11];height=[5,4,2,11,3]
    colors=[BLUE,ORANGE,GREEN,GREEN,PALE]
    for i,(b,h,col) in enumerate(zip(bottom,height,colors)):
        ax.bar(i,h,bottom=b,width=.63,color=col,edgecolor=RED if i==4 else col,hatch='///' if i==4 else None,zorder=3)
        ax.text(i,b+h/2,['5','+4','+2','11','3'][i],ha='center',va='center',fontsize=25,fontweight='bold',color=INK if i==4 else 'white')
    for i,h in enumerate([5,9,11]):ax.plot([i+.32,i+.68],[h,h],color=GRAY,ls=':',lw=1.5)
    ax.text(1,9.45,'9 total',ha='center',color=ORANGE,fontweight='bold');ax.text(2,11.45,'11 total',ha='center',color=GREEN,fontweight='bold')
    ax.axhline(14,color=GRAY,ls='--',lw=1);ax.text(-.4,14.25,'14 candidates',color=GRAY)
    ax.set_xticks(pos,['Base PySR','Added by\nSRBench bundle','Added by\nMIPS evolution','Total\nrecovered','Still\nunsolved'])
    ax.set_ylim(0,15.5);ax.set_yticks([0,5,9,11,14]);ax.set_ylabel('Previously unsolved problems');ax.grid(axis='y',alpha=.1)
    finish(fig,'02_waterfall')


def donuts():
    fig=setup('The 14 SR candidates: 5 → 9 → 11 solved')
    ax=fig.add_axes([.05,.20,.47,.56]);ax.set_aspect('equal')
    # Identical total in each ring preserves exact hierarchical alignment.
    rings=[([30,32],[GRAY,PALE],.94),([30,18,14],[GRAY,'#c9d1d7',BLUE],1.17),([30,18,5,4,2,3],[GRAY,'#c9d1d7',BLUE,ORANGE,GREEN,RED],1.40)]
    for vals,cols,r in rings:ax.pie(vals,radius=r,colors=cols,startangle=90,counterclock=False,wedgeprops=dict(width=.21,edgecolor='white'))
    ax.text(0,.09,'62',ha='center',fontsize=34,fontweight='bold',color=INK);ax.text(0,-.2,'problems',ha='center',color=GRAY)
    ax.set_xlim(-1.65,1.65);ax.set_ylim(-1.55,1.55)
    for i,(col,label) in enumerate([(GRAY,'30 originally solved'),('#c9d1d7','18 excluded from SR'),(BLUE,'5 solved by base PySR'),(ORANGE,'+4 added by SRBench bundle'),(GREEN,'+2 added by MIPS evolution'),(RED,'3 remain unsolved')]):
        yy=.72-i*.064;fig.add_artist(Rectangle((.57,yy-.008),.016,.019,transform=fig.transFigure,color=col));fig.text(.60,yy,label,fontsize=13,va='center',color=INK)
    fig.text(.57,.255,'5 → 9 → 11',fontsize=36,fontweight='bold',color=GREEN)
    fig.text(.57,.205,'Cumulative solves among 14 candidates',fontsize=12,color=INK)
    fig.text(.065,.135,'Inner: solved / unsolved    Middle: SR suitability    Outer: first method to solve',fontsize=10,color=GRAY)
    finish(fig,'03_nested_donut')


def sankey():
    fig=setup('Where the gains come from',context=False)
    ax=fig.add_axes([.035,.19,.94,.64]);ax.set_xlim(-1,10.8);ax.set_ylim(-3,73);ax.axis('off')
    def ribbon(x0,y0,x1,y1,h,color):
        mid=(x0+x1)/2
        verts=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+h),(mid,y1+h),(mid,y0+h),(x0,y0+h),(x0,y0)]
        ax.add_patch(PathPatch(MPath(verts,[1,4,4,4,2,4,4,4,79]),facecolor=color,alpha=.30,edgecolor='none'))
    def node(x,y,h,c,label,side='right'):
        ax.add_patch(Rectangle((x,y),.12,h,color=c));ax.text(x+.22 if side=='right' else x-.12,y+h/2,label,ha='left' if side=='right' else 'right',va='center',fontsize=11,color=INK)
    # All ribbons share one scale: height equals problem count.
    ribbon(.12,36,2,40,30,GRAY);ribbon(.12,4,2,2,32,BLUE)
    ribbon(2.12,16,4.6,26,18,GRAY);ribbon(2.12,2,4.6,2,14,BLUE)
    for y0,y1,h,col in [(11,18,5,BLUE),(7,12,4,ORANGE),(5,8,2,GREEN),(2,2,3,RED)]:ribbon(4.72,y0,8,y1,h,col)
    node(0,4,62,INK,'62\nproblems',side='left')
    node(2,40,30,GRAY,'30 originally\nsolved');node(2,2,32,BLUE,'32 unsolved',side='left')
    node(4.6,26,18,GRAY,'18 excluded from SR');node(4.6,2,14,BLUE,'14 candidates',side='left')
    for y,h,col,label in [(18,5,BLUE,'5  Base PySR'),(12,4,ORANGE,'+4  SRBench bundle'),(8,2,GREEN,'+2  MIPS evolution'),(2,3,RED,'3  Still unsolved')]:node(8,y,h,col,label)
    ax.text(6.4,61,'5 → 9 → 11',fontsize=33,fontweight='bold',color=GREEN)
    ax.text(6.4,54,'Cumulative solves among\nthe 14 SR candidates',fontsize=12,color=INK)
    fig.text(.065,.135,'Ribbon width = number of problems. Recovery branches show the first method to solve each problem.',fontsize=10,color=GRAY)
    finish(fig,'04_sankey')


def paired(results):
    fig=setup('Recovery improves at both problem and subtask level')
    for j,(vals,total,title) in enumerate([([5,9,11],14,'Previously unsolved overall problems'),
        ([r['scalar_solved_any_seed'] for r in results.values()],51,'Distinct SR subtasks (entire evaluation split)')]):
        ax=fig.add_axes([.08+j*.48,.25,.40,.48])
        ax.step([0,1,2,2.15],vals+[vals[-1]],where='post',color=INK,lw=2)
        ax.axhline(total,color=GRAY,ls='--',lw=1)
        for x,(y,col) in enumerate(zip(vals,[BLUE,ORANGE,GREEN])):
            ax.scatter(x,y,s=95,color=col,zorder=3)
            ax.text(x,y+total*.06,f'{y}/{total}',ha='center',color=col,fontweight='bold',fontsize=17)
        ax.set_ylim(0,total*1.12);ax.set_xlim(-.25,2.4)
        ax.set_xticks([0,1,2],['Base PySR','SRBench\nbundle','MIPS\nevolution'])
        ax.set_title(title,loc='left',fontsize=12,pad=20,fontweight='bold')
        ax.grid(axis='y',alpha=.1)
    fig.text(.065,.135,'Subtasks: solved at least once in 10 seeds; 51 distinct subtasks × 10 seeds = 510 fits per method.',fontsize=10,color=GRAY)
    finish(fig,'05_problems_and_subtasks')


def main():
    results=data()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42})
    staircase();waterfall();donuts();sankey();paired(results)
    fig,axs=plt.subplots(2,2,figsize=(16,9.33))
    for ax,name in zip(axs.flat,['01_staircase','02_waterfall','03_nested_donut','04_sankey']):ax.imshow(plt.imread(OUT/f'{name}.png'));ax.axis('off')
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=.01,hspace=.01);fig.savefig(OUT/'overview.png',dpi=150);plt.close(fig)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='group_exact_seeds'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
