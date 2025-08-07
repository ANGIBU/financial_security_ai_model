# manual_correction.py
"""
수동 교정
"""

import json
import csv
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass 
class CorrectionEntry:
    """교정 항목"""
    question_id: str
    question: str
    predicted_answer: str
    corrected_answer: str
    correction_reason: str
    confidence_boost: float

class ManualCorrectionSystem:
    """수동 교정 시스템"""
    
    def __init__(self):
        self.corrections = []
        self.correction_patterns = {}
        self.correction_stats = {
            "total_corrections": 0,
            "mc_corrections": 0,
            "subj_corrections": 0
        }
        
    def add_correction(self, question: str, predicted: str, 
                      correct: str, reason: str = "",
                      question_id: str = None) -> None:
        """교정 추가"""
        
        if question_id is None:
            question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # 신뢰도 부스트 계산
        if predicted == correct:
            confidence_boost = 0.1
        else:
            confidence_boost = 0.3
        
        entry = CorrectionEntry(
            question_id=question_id,
            question=question,
            predicted_answer=predicted,
            corrected_answer=correct,
            correction_reason=reason,
            confidence_boost=confidence_boost
        )
        
        self.corrections.append(entry)
        self.correction_stats["total_corrections"] += 1
        
        # 객관식/주관식 구분
        if correct.isdigit() and 1 <= int(correct) <= 5:
            self.correction_stats["mc_corrections"] += 1
        else:
            self.correction_stats["subj_corrections"] += 1
        
        # 패턴 학습
        self._learn_correction_pattern(entry)
    
    def _learn_correction_pattern(self, entry: CorrectionEntry) -> None:
        """교정 패턴 학습"""
        
        # 키워드 추출
        keywords = self._extract_key_phrases(entry.question)
        
        for keyword in keywords:
            if keyword not in self.correction_patterns:
                self.correction_patterns[keyword] = {}
            
            if entry.corrected_answer not in self.correction_patterns[keyword]:
                self.correction_patterns[keyword][entry.corrected_answer] = 0
            
            self.correction_patterns[keyword][entry.corrected_answer] += 1
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """핵심 구문 추출"""
        
        import re
        phrases = []
        
        # 법령 표현
        laws = re.findall(r'\w+법', text.lower())
        phrases.extend(laws)
        
        # 핵심 용어
        key_terms = ["개인정보", "전자금융", "암호화", "보안", "관리체계"]
        for term in key_terms:
            if term in text.lower():
                phrases.append(term)
        
        # 부정형 표현
        if "해당하지" in text or "적절하지" in text:
            phrases.append("부정형")
        
        return phrases
    
    def apply_corrections(self, question: str, predicted: str) -> Tuple[str, float]:
        """교정 적용"""
        
        # 직접 매칭
        for correction in self.corrections:
            if correction.question == question:
                return correction.corrected_answer, 0.9
        
        # 패턴 기반 교정
        keywords = self._extract_key_phrases(question)
        answer_scores = {}
        
        for keyword in keywords:
            if keyword in self.correction_patterns:
                for answer, count in self.correction_patterns[keyword].items():
                    if answer not in answer_scores:
                        answer_scores[answer] = 0
                    answer_scores[answer] += count
        
        if answer_scores:
            best_answer = max(answer_scores.items(), key=lambda x: x[1])
            confidence = min(best_answer[1] / sum(answer_scores.values()), 0.7)
            return best_answer[0], confidence
        
        # 교정 없음
        return predicted, 0.0
    
    def load_corrections_from_csv(self, filepath: str) -> int:
        """CSV에서 교정 데이터 로드"""
        
        if not os.path.exists(filepath):
            return 0
        
        loaded = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.add_correction(
                        question=row.get('question', ''),
                        predicted=row.get('predicted', ''),
                        correct=row.get('correct', ''),
                        reason=row.get('reason', ''),
                        question_id=row.get('id')
                    )
                    loaded += 1
        except Exception as e:
            print(f"CSV 로드 오류: {e}")
        
        return loaded
    
    def save_corrections_to_csv(self, filepath: str = "./corrections.csv") -> None:
        """교정 데이터 CSV 저장"""
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'question', 'predicted', 'correct', 'reason']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for correction in self.corrections:
                writer.writerow({
                    'id': correction.question_id,
                    'question': correction.question[:200],
                    'predicted': correction.predicted_answer,
                    'correct': correction.corrected_answer,
                    'reason': correction.correction_reason
                })
    
    def interactive_correction(self, questions: List[Dict], 
                             predictions: List[str]) -> int:
        """대화형 교정"""
        
        corrections_made = 0
        
        print("\n=== 수동 교정 모드 ===")
        print("각 문제의 예측 답변을 확인하고 교정하세요.")
        print("Enter: 수락, 숫자/텍스트: 교정, q: 종료\n")
        
        for i, (q_data, pred) in enumerate(zip(questions, predictions)):
            print(f"\n[{i+1}/{len(questions)}]")
            print(f"문제: {q_data['question'][:150]}...")
            print(f"예측 답변: {pred}")
            
            user_input = input("교정 (Enter=수락, q=종료): ").strip()
            
            if user_input.lower() == 'q':
                break
            elif user_input:
                # 교정
                reason = input("교정 이유 (선택): ").strip()
                
                self.add_correction(
                    question=q_data['question'],
                    predicted=pred,
                    correct=user_input,
                    reason=reason,
                    question_id=q_data.get('id')
                )
                corrections_made += 1
                print("✓ 교정 저장됨")
        
        return corrections_made
    
    def batch_correction(self, correction_file: str) -> int:
        """배치 교정"""
        
        if not os.path.exists(correction_file):
            print(f"파일 없음: {correction_file}")
            return 0
        
        corrections_made = 0
        
        try:
            with open(correction_file, 'r', encoding='utf-8') as f:
                corrections_data = json.load(f)
            
            for item in corrections_data:
                self.add_correction(
                    question=item['question'],
                    predicted=item['predicted'],
                    correct=item['correct'],
                    reason=item.get('reason', ''),
                    question_id=item.get('id')
                )
                corrections_made += 1
        
        except Exception as e:
            print(f"배치 교정 오류: {e}")
        
        return corrections_made
    
    def get_correction_stats(self) -> Dict:
        """교정 통계"""
        
        return {
            "total": self.correction_stats["total_corrections"],
            "mc": self.correction_stats["mc_corrections"],
            "subjective": self.correction_stats["subj_corrections"],
            "patterns": len(self.correction_patterns),
            "pattern_distribution": {
                pattern: len(answers) 
                for pattern, answers in self.correction_patterns.items()
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.corrections:
            self.save_corrections_to_csv()
        print(f"교정 시스템: {len(self.corrections)}개 교정 저장")