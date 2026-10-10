"""Read-only supervisory/source audit and tables for the accepted ALOHA sweep."""
from pathlib import Path
import csv,datetime,hashlib,json,statistics
ROOT=Path(__file__).parent;STAGE=ROOT.parent/'reference-full-sweep'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((ROOT/'summary.json').read_text())
assert summary['accepted_reports']==36 and summary['accepted_pairs']==18
manifest=json.loads((STAGE/'launch-source-manifest.json').read_text())
for f in manifest['files']:
    p=STAGE/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
plan=json.loads((STAGE/'plan.json').read_text());status=json.loads((STAGE/'plan.status.json').read_text())
assert status['status']=='finished' and len(status['jobs'])==len(plan['jobs'])==36
assert status['plan_sha256']==sha(STAGE/'plan.json')
previous=None
for expected,actual in zip(plan['jobs'],status['jobs'],strict=True):
    assert expected['id']==actual['id'] and expected['command']==actual['command']
    assert actual['returncode']==0 and actual['status']=='passed' and actual['report_status']=='complete'
    start=datetime.datetime.fromisoformat(actual['started_utc']);end=datetime.datetime.fromisoformat(actual['finished_utc'])
    assert end>=start and (previous is None or start>=previous)
    previous=end
reports={};health=[];episode_rows=[]
for row in summary['reports']:
    path=Path(row['path']);d=json.loads(path.read_text());key=(row['stack'],row['backend'],row['worlds']);reports[key]=d
    assert len(d['warmups'])==1 and len(d['samples'])==5 and d['steps']==1001 and d['dt']==.002
    for kind,items in [('warmup',d['warmups']),('measured',d['samples'])]:
        for i,s in enumerate(items):
            v=s['validation'];assert v['passed'] and all(v['checks'].values())
            diag=v['physical_health']['diagnostics'];last=diag['checkpoints'][-1]
            health.append({'stack':key[0],'backend':key[1],'worlds':key[2],'kind':kind,'episode':i,
                **{k:diag[k] for k in ['max_quaternion_norm_error','max_free_translation_excursion_m','max_abs_qvel','max_velocity_bound_fraction']},
                **{k:last[k] for k in ['pot_z_min','pot_z_max','lid_z_min','lid_z_max']},
                'capacity':{k:v for k,v in s.get('diagnostics',{}).items() if k!='overflow'}})
            episode_rows.append({'stack':key[0],'backend':key[1],'worlds':key[2],'kind':kind,'episode':i,
                **{k:s[k] for k in ['simulation_seconds','output_collection_seconds','history_transfer_seconds','validation_seconds']},
                'checked_seconds':s['simulation_seconds']+s['output_collection_seconds']+s['validation_seconds']})
rows=[]
for pair in summary['pairs']:
    stack,n=pair['stack'],pair['worlds'];cpu=reports[(stack,'cpu',n)];gpu=reports[(stack,'gpu',n)]
    row={'stack':stack,'worlds':n,'cpu_threads':cpu['backend_metadata']['cpu_threads'],'gpu_device':gpu['backend_metadata']['device']}
    for b,d in [('cpu',cpu),('gpu',gpu)]:
        for metric,values in [('simulation',[s['simulation_seconds'] for s in d['samples']]),('checked',[s['simulation_seconds']+s['output_collection_seconds']+s['validation_seconds'] for s in d['samples']])]:
            row[f'{b}_{metric}_median_seconds']=statistics.median(values);row[f'{b}_{metric}_min_seconds']=min(values);row[f'{b}_{metric}_max_seconds']=max(values)
        row[f'{b}_world_steps_per_second']=n*1001/row[f'{b}_simulation_median_seconds']
    row.update({k:pair[k] for k in ['simulation_speedup_cpu_over_gpu','checked_speedup_cpu_over_gpu']});rows.append(row)
def writecsv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('paired-results.csv',rows);writecsv('episode-timings.csv',episode_rows)
(ROOT/'health-summary.json').write_text(json.dumps(health,indent=2)+'\n')
inventory_path=Path('/private/tmp/blogs-box-sweep-v3-20261009/article3/smoke-so101-v3/preflight.json');inventory=json.loads(inventory_path.read_text());h=inventory['hardware']
hw={'inventory_source':str(inventory_path),'inventory_sha256':sha(inventory_path),'inventory_utc':inventory['started_utc'],
    'scope':'Retained workstation identity inventory, not a utilization trace for this reference run.',
    **{k:h[k] for k in ['cpu_model','physical_cpu_cores','logical_cpu_count','ram_bytes','os','os_release','architecture']},
    'selected_gpu':'cuda:1','gpu_name':reports[('blog2','gpu',1)]['backend_metadata']['device_name'],
    'gpu_inventory':[{'index':g['index'],'name':g['name'],'driver':g['driver'],'memory_total_mib':g['memory_total_mib']} for g in h['load']['gpus']],
    'worker_policy':'min(worlds,32,available affinity):1,16,then32 native C++ rollout threads',
    'shared_workstation':True,'concurrent_utilization_trace_available':False}
(ROOT/'hardware.json').write_text(json.dumps(hw,indent=2)+'\n')
receipt={'accepted_reports':36,'accepted_pairs':18,'measured_batch_episodes':180,'warmup_batch_episodes':36,
    'supervisor_started_utc':status['started_utc'],'supervisor_finished_utc':status['finished_utc'],'supervisor_jobs_sequential':True,
    'all_source_manifest_files_verified':len(manifest['files']),'retained_failed_cases':0,
    'coarse_health_all_episodes_passed':True,'timing_eligibility':'All36cases meet frozen collector1warm5measured gates; no failures to exclude in this reference sweep.',
    'limitations':summary['limitations']+['No contemporaneous utilization trace; workstation is shared.','Five repeated fixed-tape episodes do not establish statistical confidence or randomized-task robustness.','Reference is native MuJoCo versus MuJoCo Warp, including on Blog3 stack; it is not Newton API timing.'],
    'source_files_sha256':{f:sha(STAGE/f) for f in ['launch-source-manifest.json','plan.json','plan.status.json','launch-receipt.json','supervisor.py']},
    'audit_artifacts_sha256':{f:sha(ROOT/f) for f in ['summary.json','paired-results.csv','episode-timings.csv','health-summary.json','hardware.json','package_audit.py']}}
(ROOT/'audit-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
