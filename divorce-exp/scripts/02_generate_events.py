"""
Critical Events Generator for Divorce Prediction

Generates personalized stress-test scenarios based on Gottman DPS profiles.
Each couple faces 3 critical events:
1. Marriage Milestone (commitment test)
2. Trust Breach (forgiveness test)  
3. Illness/Caregiver (endurance test)

These events surface true feelings that predict divorce better than stated values.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


class CriticalEventsGenerator:
    """Generate personalized critical events based on Gottman profile."""
    
    def __init__(self, clean_data_path: str = "divorce_clean.csv"):
        """
        Load clean divorce dataset (leakage features removed).
        
        Args:
            clean_data_path: Path to cleaned CSV
        """
        self.df = pd.read_csv(clean_data_path)
        print(f"✅ Loaded {len(self.df)} couples")
        print(f"   Features: {self.df.shape[1] - 1} (excluding target)")
        print(f"   Divorced: {(self.df['Class']==0).sum()}, Married: {(self.df['Class']==1).sum()}")
        
        # Map attribute numbers to question text (from doc.txt)
        self.question_map = self._load_question_map()
        
    def _load_question_map(self) -> Dict[str, str]:
        """Map Atr1-Atr54 to actual questions."""
        # Shortened for brevity - full mapping in doc.txt
        return {
            'Atr1': 'Apologizes when discussions go bad',
            'Atr2': 'Can ignore differences even when hard',
            'Atr3': 'Can restart and correct discussions',
            'Atr4': 'Contacting eventually works',
            'Atr5': 'Time spent together is special',
            'Atr6': 'Have time at home as partners',
            'Atr7': 'Not like strangers sharing environment',
            'Atr8': 'Enjoy holidays together',
            'Atr9': 'Enjoy traveling together',
            'Atr10': 'Most goals are common',
            'Atr11': 'See future harmony together',
            'Atr12': 'Similar values in personal freedom',
            'Atr13': 'Similar entertainment preferences',
            'Atr14': 'Same goals for children/friends',
            'Atr15': 'Similar dreams about life',
            'Atr16': 'Compatible views on love',
            'Atr17': 'Similar values in trust',
            'Atr18': 'Share views on happiness',
            'Atr19': 'Similar ideas about marriage',
            'Atr20': 'Similar ideas about roles in marriage',
            # ... (add full 54 questions from doc.txt)
        }
    
    def generate_events_for_couple(self, couple_id: int) -> Dict:
        """
        Generate 3 personalized critical events for a couple.
        
        Args:
            couple_id: Row index in dataframe
            
        Returns:
            Dict with couple profile + 3 critical events
        """
        row = self.df.iloc[couple_id]
        
        # Extract key profile attributes
        profile = {
            'couple_id': couple_id,
            'ground_truth_divorced': int(row['Class'] == 0),
            'trust': int(row.get('Atr17', 2)),  # Default to middle if removed
            'shared_goals': int(row.get('Atr10', 2)),
            'personal_freedom': int(row.get('Atr12', 2)),
            'conflict_resolution': int(row.get('Atr3', 2)),
            'forgiveness': int(row.get('Atr1', 2)),
            'time_together': int(row.get('Atr5', 2)),
            'compatibility': int(row.get('Atr16', 2)),
        }
        
        # Generate 3 critical events
        events = {
            'event1_marriage_milestone': self._generate_marriage_event(profile),
            'event2_trust_breach': self._generate_trust_breach_event(profile),
            'event3_illness_burden': self._generate_illness_event(profile),
        }
        
        return {
            'profile': profile,
            'events': events,
        }
    
    def _generate_marriage_event(self, profile: Dict) -> Dict:
        """
        Generate marriage milestone event (commitment test).
        Personalized based on shared_goals score.
        """
        # High shared goals → test with career choice
        # Low shared goals → test with major life decision
        
        if profile['shared_goals'] >= 3:
            # They claim shared goals - test it
            scenario = {
                'type': 'career_vs_family',
                'description': (
                    "Your spouse receives their dream job offer that would require "
                    "relocating 2000 miles away from your current city. This means:\n"
                    "- You must quit your job (no remote option)\n"
                    "- Leave your family, friends, and support network\n"
                    "- Start over in a new city where you know nobody\n\n"
                    "Your spouse is VERY excited and sees this as 'our big opportunity.'"
                ),
                'decision_point': (
                    "Your spouse asks: 'We're doing this together, right? This is what we always talked about.'\n\n"
                    "Do you: (A) Support the move, (B) Ask them to decline, (C) Suggest long-distance"
                ),
            }
        else:
            # They admit goals differ - test breaking point
            scenario = {
                'type': 'children_timing',
                'description': (
                    "You and your spouse are at dinner. They suddenly say:\n"
                    "'I want to start trying for a baby next month. I've been thinking about this "
                    "for months and I'm ready. I know we said 'someday' but someday is now.'\n\n"
                    "Problem: You want to wait at least 3-5 more years (career, finances, freedom)."
                ),
                'decision_point': (
                    "Your spouse says: 'If you love me, you'll do this with me. This is non-negotiable.'\n\n"
                    "Do you: (A) Agree to start trying, (B) Hold your ground (wait), (C) Suggest therapy/compromise"
                ),
            }
        
        scenario['profile_anchor'] = f"Your 'shared goals' score: {profile['shared_goals']}/4"
        return scenario
    
    def _generate_trust_breach_event(self, profile: Dict) -> Dict:
        """
        Generate trust breach event (forgiveness test).
        Personalized based on trust score.
        """
        if profile['trust'] >= 3:
            # High trust → bigger betrayal needed to test forgiveness
            scenario = {
                'type': 'emotional_affair',
                'description': (
                    "You find intimate text messages on your spouse's phone with a coworker:\n\n"
                    "- 'You understand me in ways my spouse never will'\n"
                    "- 'I think about you all day'\n"
                    "- 'Meeting you was the best thing that happened this year'\n"
                    "- Plans to meet for drinks (which they told you was 'working late')\n\n"
                    "When confronted, your spouse says: 'It's not physical! We're just friends. "
                    "You're overreacting. Nothing happened.'"
                ),
                'betrayal_severity': 7,  # 0-10 scale
            }
        else:
            # Low trust → even small betrayal confirms fears
            scenario = {
                'type': 'financial_deception',
                'description': (
                    "While doing taxes, you discover your spouse has been hiding purchases:\n"
                    "- $12,000 spent on online gambling over 8 months\n"
                    "- Money taken from joint savings (the 'house down payment' fund)\n"
                    "- Secret credit card with $8,000 balance\n\n"
                    "When confronted: 'I was going to win it back. I didn't want to worry you. "
                    "It's my money too, I can spend it how I want.'"
                ),
                'betrayal_severity': 8,
            }
        
        scenario['profile_anchor'] = f"Your 'trust' score: {profile['trust']}/4"
        scenario['forgiveness_score'] = profile['forgiveness']
        return scenario
    
    def _generate_illness_event(self, profile: Dict) -> Dict:
        """
        Generate illness/caregiver event (endurance test).
        Personalized based on personal_freedom and time_together scores.
        """
        if profile['personal_freedom'] >= 3 and profile['time_together'] <= 2:
            # Values freedom + doesn't prioritize togetherness → worst case scenario
            scenario = {
                'type': 'chronic_pain_full_care',
                'description': (
                    "Your spouse develops chronic migraines (no cure, only management):\n"
                    "- They can't work anymore (lost income)\n"
                    "- Need help with basic tasks when migraines hit (3-4 days/week)\n"
                    "- Can't handle noise, light, or social activities\n"
                    "- Your life becomes: work → silent house → caregiver → exhausted → repeat\n\n"
                    "Your hobbies, friends, and 'me time' disappear. This is permanent."
                ),
                'care_hours_per_week': 35,
                'duration': 'indefinite',
            }
        elif profile['time_together'] >= 3:
            # Values togetherness → test quality vs quantity
            scenario = {
                'type': 'depression_withdrawal',
                'description': (
                    "Your spouse develops severe depression:\n"
                    "- They stop talking to you (monosyllabic responses)\n"
                    "- No physical intimacy for 6+ months\n"
                    "- Refuse to go to therapy ('therapists don't help')\n"
                    "- Sleep 14+ hours/day, don't do household tasks\n\n"
                    "You live with a ghost. The person you married is 'gone' but their body is still there."
                ),
                'care_hours_per_week': 20,
                'duration': 'unknown (they refuse treatment)',
            }
        else:
            # Balanced → standard caregiver burden
            scenario = {
                'type': 'post_accident_disability',
                'description': (
                    "Your spouse is in a car accident (not at fault):\n"
                    "- Permanent lower-body paralysis (wheelchair-bound)\n"
                    "- Need help with bathing, dressing, transfers\n"
                    "- Can't drive, need rides to medical appointments (3x/week)\n"
                    "- Angry and resentful about their new reality (takes it out on you)\n\n"
                    "Doctor says: 'This is your life now. Adapt or burn out.'"
                ),
                'care_hours_per_week': 25,
                'duration': 'lifelong',
            }
        
        scenario['profile_anchor'] = (
            f"Your 'personal freedom' score: {profile['personal_freedom']}/4\n"
            f"Your 'time together importance' score: {profile['time_together']}/4"
        )
        return scenario
    
    def generate_all_couples(self, output_path: str = "critical_events.json"):
        """
        Generate critical events for all couples in dataset.
        
        Args:
            output_path: Where to save JSON output
        """
        print(f"\n🎭 Generating critical events for {len(self.df)} couples...")
        
        all_events = []
        for i in range(len(self.df)):
            couple_events = self.generate_events_for_couple(i)
            all_events.append(couple_events)
            
            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{len(self.df)} couples...")
        
        # Save to JSON
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_events, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved critical events to: {output_path}")
        print(f"   Total couples: {len(all_events)}")
        print(f"   Events per couple: 3")
        print(f"   Total events: {len(all_events) * 3}")
        
        # Show sample
        print("\n📖 Sample event (Couple 0, Marriage Milestone):")
        print(json.dumps(all_events[0]['events']['event1_marriage_milestone'], indent=2))
        
        return all_events


def main():
    """Generate critical events for all couples."""
    print("=" * 70)
    print("CRITICAL EVENTS GENERATOR")
    print("=" * 70)
    
    # Initialize generator
    generator = CriticalEventsGenerator("divorce_clean.csv")
    
    # Generate all events
    all_events = generator.generate_all_couples("critical_events.json")
    
    print("\n🎯 Next steps:")
    print("   1. Review critical_events.json")
    print("   2. Run 03_llm_simulator.py to simulate responses")
    print("   3. Evaluate predictions vs ground truth")


if __name__ == "__main__":
    main()
