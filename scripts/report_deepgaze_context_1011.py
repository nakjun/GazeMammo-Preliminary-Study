"""Combine complete fixed DeepGaze context benchmarks without partial-result claims."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'runs/dual_target_1011/deepgaze_public';results=[]
    for name in ['val93','kde256_val93','kde512_val93']:
        value=json.loads((directory/name/'metrics.json').read_text())
        if not value['complete'] or value['images']!=93:raise ValueError('All three fixed93 runs must complete')
        results.append(dict(run=name,nll=value['nll'],seconds=value['seconds'],input_size=value.get('input_size',[256,512]),centerbias=value.get('centerbias','uniform'),proofs=value['proofs']))
    selected=min(results[1:],key=lambda x:x['nll']);output=directory/'context_comparison';output.mkdir(exist_ok=False);(output/'metrics.json').write_text(json.dumps(dict(results=results,selected_fixed_context=selected,selection='Two fixed resolutions, exploratory val93 choice; four-timespixels if512selected; no independenttest claim'),indent=2));pd.DataFrame([{k:v for k,v in r.items() if k!='proofs'} for r in results]).to_csv(output/'comparison.csv',index=False)
    lines=['# DeepGaze fixed context comparison','','Frozen public natural-gaze model, fixed val93 canonical16/grid24×48, actual image-only arithmetic BOS. Two prespecified input resolutions and one train-only KDE; no temperature/sigma sweep or fine-tuning.','','| Context | NLL | Seconds |','|---|---:|---:|'];lines += [f"| {r['run']} | {r['nll']:.6f} | {r['seconds']:.2f} |" for r in results];lines += ['',f"Chosen development context: {selected['input_size']}, train-only KDE. Validation choice is exploratory. 512×1024 costs4× whole-image pixels versus256×512; target/grid/224 physicalcrop budget unchanged.",'','KDE uses exact originaltrain412 canonicalmaskedcoordinates with imageuniformweights, Gaussianσ1coarsecell, reflectboundary, positivefloor1e−8. Probability interpolatedbilinearly toinputpixels, thenrenormalizedandlogged. Validation gaze never fits this prior. Source hashes,327trainpatients and412IDs saved in ../train_centerbias_provenance.json.','','Native DenseNet featuretuple1888/128/32channels at8×4 or16×8; cache retainsnativearrays, no artificialcached upsampling. Two realimages eachresolutioncached/live logdensitymax0; futurehistoryinputsexactequal, repeatedGPUoutputerror≤2.86e−6. Probabilitynormalized; upstreamemptyhistoryfailurehandledbyactualimage-onlyBOSarm. Helpertests2PASS.','','NLL5.3 remains unmet. These frozen transfer results motivate a separately authorized train412-only readout experiment; they do not establish clinical benefit. License unresolved/localresearch only. Naturalpretraining SALICONmouseproxy andMIT1003, source [authorspaper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/).'];(output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(selected))

if __name__=='__main__':main()
