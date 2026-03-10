#!/usr/bin/env python
"""
Qualitative Sampler for extracting key reasoning examples
Implements Step 3.3 of Phase 3
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import re


class QualitativeSampler:
    def __init__(self, data_file: str = "analysis/unified_consolidated_data.jsonl"):
        """Initialize with unified consolidated data"""
        self.data_file = Path(data_file)
        self.records = []
        self.pilot_data = []
        self.curriculum_data = []
        self.load_data()
        
        # Categories for pilot study PGG reasoning
        self.pgg_categories = {
            'RATIONAL_CHOICE': ['maximize', 'rational', 'optimal', 'best response', 'nash', 'dominant'],
            'GREED': ['keep', 'profit', 'gain', 'benefit myself', 'selfish', 'more for me'],
            'FEAR': ['others might', 'risky', 'afraid', 'uncertain', 'worry', 'concerned'],
            'LACK_OF_TRUST': ['trust', 'distrust', 'betray', 'reliable', 'faith', 'confidence'],
            'CONFUSION': ['unclear', 'confused', 'understand', 'not sure', 'uncertain about rules']
        }
    
    def load_data(self):
        """Load and separate pilot and curriculum data"""
        with open(self.data_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.records.append(record)
                
                if record.get('experiment_type') == 'curriculum':
                    self.curriculum_data.append(record)
                elif record.get('experiment_type') == 'pilot':
                    self.pilot_data.append(record)
        
        print(f"Loaded {len(self.records)} total records")
        print(f"  - Pilot: {len(self.pilot_data)}")
        print(f"  - Curriculum: {len(self.curriculum_data)}")
    
    def extract_key_examples(self) -> Dict[str, List[Dict]]:
        """Extract representative reasoning traces (Step 3.3.b)"""
        examples = {
            'pilot_study': {
                'pgg_rational_vs_greed': [],
                'stag_hunt_coordination': [],
                'communication_signals': []
            },
            'curriculum_study': {
                'control_vs_full_comparison': [],
                'learning_progression': [],
                'punishment_reasoning': []
            },
            'surprising_findings': []
        }
        
        # 1. Extract Pilot Study Examples
        examples['pilot_study']['pgg_rational_vs_greed'] = self._extract_pgg_reasoning_examples()
        examples['pilot_study']['stag_hunt_coordination'] = self._extract_stag_hunt_examples()
        examples['pilot_study']['communication_signals'] = self._extract_communication_examples()
        
        # 2. Extract Curriculum Study Examples
        examples['curriculum_study']['control_vs_full_comparison'] = self._compare_control_vs_full()
        examples['curriculum_study']['learning_progression'] = self._extract_learning_progression()
        examples['curriculum_study']['punishment_reasoning'] = self._extract_punishment_reasoning()
        
        # 3. Extract Surprising Findings
        examples['surprising_findings'] = self._extract_surprising_findings()
        
        return examples
    
    def _extract_pgg_reasoning_examples(self) -> List[Dict]:
        """Extract examples of RATIONAL_CHOICE vs GREED reasoning in PGG"""
        examples = []
        
        pgg_records = [r for r in self.pilot_data if 'public_goods' in r.get('game_id', '')]
        
        rational_examples = []
        greed_examples = []
        
        for record in pgg_records:
            for round_data in record.get('rounds_data', []):
                rationales = round_data.get('rationales', {})
                
                for agent, rationale in rationales.items():
                    if not rationale:
                        continue
                    
                    rationale_lower = rationale.lower()
                    
                    # Check for RATIONAL_CHOICE
                    if any(keyword in rationale_lower for keyword in self.pgg_categories['RATIONAL_CHOICE']):
                        rational_examples.append({
                            'game': record['game_id'],
                            'setting': record['setting'],
                            'trial': record['trial_id'],
                            'round': round_data['round'],
                            'agent': agent,
                            'action': round_data.get('contributions', {}).get(agent, 0),
                            'rationale': rationale,
                            'category': 'RATIONAL_CHOICE'
                        })
                    
                    # Check for GREED
                    elif any(keyword in rationale_lower for keyword in self.pgg_categories['GREED']):
                        greed_examples.append({
                            'game': record['game_id'],
                            'setting': record['setting'],
                            'trial': record['trial_id'],
                            'round': round_data['round'],
                            'agent': agent,
                            'action': round_data.get('contributions', {}).get(agent, 0),
                            'rationale': rationale,
                            'category': 'GREED'
                        })
        
        # Sample diverse examples
        if rational_examples:
            examples.append(random.choice(rational_examples))
        if greed_examples:
            examples.append(random.choice(greed_examples))
        
        # Add one contrasting pair if possible
        if rational_examples and greed_examples:
            examples.append({
                'type': 'contrast',
                'rational': rational_examples[0],
                'greed': greed_examples[0]
            })
        
        return examples[:5]  # Return up to 5 examples
    
    def _extract_stag_hunt_examples(self) -> List[Dict]:
        """Extract examples of successful vs failed coordination in Stag Hunt"""
        examples = []
        
        sh_records = [r for r in self.pilot_data if 'stag_hunt' in r.get('game_id', '')]
        
        successful = []
        failed = []
        
        for record in sh_records:
            for round_data in record.get('rounds_data', []):
                choices = round_data.get('choices', {})
                rationales = round_data.get('rationales', {})
                
                if len(choices) >= 2:
                    all_stag = all(choice.lower() == 'stag' for choice in choices.values())
                    all_hare = all(choice.lower() == 'hare' for choice in choices.values())
                    
                    for agent, rationale in rationales.items():
                        if not rationale:
                            continue
                        
                        example = {
                            'game': record['game_id'],
                            'trial': record['trial_id'],
                            'round': round_data['round'],
                            'agent': agent,
                            'choice': choices.get(agent, 'unknown'),
                            'rationale': rationale,
                            'outcome': 'successful' if all_stag else ('failed' if all_hare else 'mixed')
                        }
                        
                        if all_stag:
                            successful.append(example)
                        elif all_hare:
                            failed.append(example)
        
        # Sample examples
        if successful:
            examples.append(random.choice(successful))
        if failed:
            examples.append(random.choice(failed))
        
        return examples[:3]
    
    def _extract_communication_examples(self) -> List[Dict]:
        """Extract examples of communication signals that led to coordination"""
        examples = []
        
        comm_records = [r for r in self.pilot_data if 'communication' in r.get('game_id', '')]
        
        effective_signals = []
        
        for record in comm_records:
            for round_data in record.get('rounds_data', []):
                communications = round_data.get('communications', {})
                choices = round_data.get('choices', {})
                
                if communications and choices:
                    all_cooperated = round_data.get('all_cooperated', False)
                    
                    for agent, message in communications.items():
                        if message and all_cooperated:
                            effective_signals.append({
                                'trial': record['trial_id'],
                                'round': round_data['round'],
                                'agent': agent,
                                'message': message,
                                'choice': choices.get(agent, 'unknown'),
                                'outcome': 'successful coordination'
                            })
        
        # Find messages with "stag", "together", "unity", etc.
        key_signals = ['stag', 'together', 'unity', 'trust', 'cooperate']
        
        for signal in key_signals:
            matching = [e for e in effective_signals if signal in e['message'].lower()]
            if matching:
                examples.append(random.choice(matching))
                if len(examples) >= 3:
                    break
        
        return examples
    
    def _compare_control_vs_full(self) -> List[Dict]:
        """Compare reasoning between control and full curriculum agents in final stage"""
        examples = []
        
        # Get final stage data for control group
        control_final = [r for r in self.curriculum_data 
                        if r.get('curriculum_condition') == 'control_group']
        
        # Get final stage data for full curriculum (stage 4)
        full_final = [r for r in self.curriculum_data 
                     if r.get('curriculum_condition') == 'full_curriculum' 
                     and r.get('stage_num') == 4]
        
        # Extract rationales from each
        control_rationales = []
        full_rationales = []
        
        for record in control_final[:5]:  # Sample first 5 trials
            for round_data in record.get('rounds_data', [])[:3]:  # First 3 rounds
                rationales = round_data.get('rationales', {})
                for agent, rationale in rationales.items():
                    if rationale:
                        control_rationales.append({
                            'condition': 'control',
                            'trial': record['trial_id'],
                            'round': round_data['round'],
                            'agent': agent,
                            'rationale': rationale,
                            'contribution': round_data.get('contributions', {}).get(agent, 0)
                        })
        
        for record in full_final[:5]:  # Sample first 5 trials
            for round_data in record.get('rounds_data', [])[:3]:  # First 3 rounds
                rationales = round_data.get('rationales', {})
                for agent, rationale in rationales.items():
                    if rationale:
                        full_rationales.append({
                            'condition': 'full_curriculum',
                            'trial': record['trial_id'],
                            'round': round_data['round'],
                            'agent': agent,
                            'rationale': rationale,
                            'contribution': round_data.get('contributions', {}).get(agent, 0),
                            'lessons_learned': record.get('lessons_learned', [])
                        })
        
        # Create comparison pairs
        if control_rationales and full_rationales:
            examples.append({
                'type': 'direct_comparison',
                'control': control_rationales[0] if control_rationales else None,
                'full_curriculum': full_rationales[0] if full_rationales else None
            })
        
        return examples
    
    def _extract_learning_progression(self) -> List[Dict]:
        """Extract examples showing learning progression through curriculum stages"""
        examples = []
        
        # Focus on full curriculum condition
        full_curriculum = [r for r in self.curriculum_data 
                         if r.get('curriculum_condition') == 'full_curriculum']
        
        # Group by trial
        trials = defaultdict(list)
        for record in full_curriculum:
            trials[record['trial_id']].append(record)
        
        # Sample one trial and show progression
        if trials:
            sample_trial = list(trials.values())[0]
            sample_trial.sort(key=lambda x: x.get('stage_num', 0))
            
            progression = []
            for stage_record in sample_trial:
                # Get first round rationale
                if stage_record.get('rounds_data'):
                    first_round = stage_record['rounds_data'][0]
                    rationales = first_round.get('rationales', {})
                    
                    if rationales:
                        agent, rationale = list(rationales.items())[0]
                        progression.append({
                            'stage': stage_record['stage_name'],
                            'stage_num': stage_record['stage_num'],
                            'game': stage_record['game_id'],
                            'agent': agent,
                            'rationale': rationale,
                            'action': first_round.get('contributions', first_round.get('choices', {})).get(agent),
                            'lesson': stage_record.get('lessons_learned', [])[-1] if stage_record.get('lessons_learned') else None
                        })
            
            if progression:
                examples.append({
                    'type': 'learning_progression',
                    'trial': sample_trial[0]['trial_id'],
                    'stages': progression
                })
        
        return examples
    
    def _extract_punishment_reasoning(self) -> List[Dict]:
        """Extract reasoning about punishment in final IPGG stage"""
        examples = []
        
        # Find IPGG with punishment stages
        punishment_records = [r for r in self.curriculum_data 
                            if 'public' in r.get('game_id', '').lower() 
                            and 'punishment' in r.get('stage_name', '').lower()]
        
        punishment_examples = []
        
        for record in punishment_records:
            for round_data in record.get('rounds_data', []):
                punishments = round_data.get('punishments', {})
                rationales = round_data.get('rationales', {})
                
                # Find agents who punished
                for agent, punishment_dict in punishments.items():
                    if punishment_dict and any(p > 0 for p in punishment_dict.values()):
                        if agent in rationales and rationales[agent]:
                            punishment_examples.append({
                                'condition': record['curriculum_condition'],
                                'trial': record['trial_id'],
                                'round': round_data['round'],
                                'agent': agent,
                                'punishments_given': punishment_dict,
                                'rationale': rationales[agent],
                                'contribution': round_data.get('contributions', {}).get(agent, 0)
                            })
        
        # Sample diverse examples
        if punishment_examples:
            examples.extend(random.sample(punishment_examples, min(3, len(punishment_examples))))
        
        return examples
    
    def _extract_surprising_findings(self) -> List[Dict]:
        """Extract unexpected or counter-intuitive reasoning examples"""
        surprising = []
        
        # 1. Look for sophisticated understanding
        sophisticated_keywords = ['reciprocity', 'tit-for-tat', 'conditional cooperation', 
                                'reputation', 'signaling', 'commitment', 'credible threat']
        
        for record in self.records:
            for round_data in record.get('rounds_data', []):
                rationales = round_data.get('rationales', {})
                
                for agent, rationale in rationales.items():
                    if rationale:
                        rationale_lower = rationale.lower()
                        if any(keyword in rationale_lower for keyword in sophisticated_keywords):
                            surprising.append({
                                'type': 'sophisticated_reasoning',
                                'experiment': record['experiment_type'],
                                'game': record['game_id'],
                                'agent': agent,
                                'rationale': rationale,
                                'keywords_found': [k for k in sophisticated_keywords if k in rationale_lower]
                            })
        
        # 2. Look for confused or flawed understanding
        confusion_patterns = [
            'contribute everything.*maximize.*payoff',  # Misunderstanding PGG
            'stag.*safe.*choice',  # Misunderstanding stag hunt
            'punish.*myself',  # Confusion about punishment
        ]
        
        for record in self.records:
            for round_data in record.get('rounds_data', []):
                rationales = round_data.get('rationales', {})
                
                for agent, rationale in rationales.items():
                    if rationale:
                        for pattern in confusion_patterns:
                            if re.search(pattern, rationale.lower()):
                                surprising.append({
                                    'type': 'flawed_understanding',
                                    'experiment': record['experiment_type'],
                                    'game': record['game_id'],
                                    'agent': agent,
                                    'rationale': rationale,
                                    'pattern_matched': pattern
                                })
        
        # Sample diverse surprising findings
        return random.sample(surprising, min(5, len(surprising))) if surprising else []
    
    def save_key_rationales(self, output_file: str = "analysis/key_rationales.json") -> Path:
        """Save selected traces to structured file (Step 3.3.c)"""
        output_path = Path(output_file)
        
        # Extract all key examples
        examples = self.extract_key_examples()
        
        # Create structured output
        output_data = {
            'metadata': {
                'total_records_analyzed': len(self.records),
                'pilot_records': len(self.pilot_data),
                'curriculum_records': len(self.curriculum_data),
                'extraction_categories': list(examples.keys())
            },
            'examples': examples,
            'summary': self._generate_summary(examples)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved key rationales to {output_path}")
        return output_path
    
    def _generate_summary(self, examples: Dict) -> Dict:
        """Generate summary statistics about extracted examples"""
        summary = {
            'pilot_examples_extracted': {
                'pgg_reasoning': len(examples['pilot_study']['pgg_rational_vs_greed']),
                'stag_hunt': len(examples['pilot_study']['stag_hunt_coordination']),
                'communication': len(examples['pilot_study']['communication_signals'])
            },
            'curriculum_examples_extracted': {
                'control_vs_full': len(examples['curriculum_study']['control_vs_full_comparison']),
                'learning_progression': len(examples['curriculum_study']['learning_progression']),
                'punishment': len(examples['curriculum_study']['punishment_reasoning'])
            },
            'surprising_findings': len(examples['surprising_findings']),
            'total_examples': sum([
                len(v) if isinstance(v, list) else sum(len(vv) for vv in v.values())
                for v in examples.values()
            ])
        }
        
        return summary


def main():
    """Main entry point for qualitative sampler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract key reasoning examples from experimental data")
    parser.add_argument("--data", default="analysis/unified_consolidated_data.jsonl",
                       help="Path to unified consolidated data")
    parser.add_argument("--output", default="analysis/key_rationales.json",
                       help="Output path for key rationales")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of examples per category")
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = QualitativeSampler(args.data)
    
    print("\n" + "="*60)
    print("QUALITATIVE SAMPLING")
    print("="*60)
    
    # Extract and save examples
    output_path = sampler.save_key_rationales(args.output)
    
    # Load and display summary
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    
    print("\nExamples Extracted:")
    print("-" * 40)
    
    print("\nPilot Study:")
    for key, count in summary['pilot_examples_extracted'].items():
        print(f"  - {key}: {count} examples")
    
    print("\nCurriculum Study:")
    for key, count in summary['curriculum_examples_extracted'].items():
        print(f"  - {key}: {count} examples")
    
    print(f"\nSurprising Findings: {summary['surprising_findings']} examples")
    print(f"\nTotal Examples Extracted: {summary['total_examples']}")
    
    print("\n" + "="*60)
    print("Key rationales successfully extracted!")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()