# manual_correction.py
"""
수동 교정
"""

import json
import csv
import os
import re
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
    korean_quality: float = 0.0

class ManualCorrectionSystem:
    """수동 교정 시스템 - 한국어 특화"""
    
    def __init__(self):
        self.corrections = []
        self.correction_patterns = {}
        self.correction_stats = {
            "total_corrections": 0,
            "mc_corrections": 0,
            "subj_corrections": 0,
            "korean_improvements": 0
        }
        
        # 한국어 교정 패턴
        self.korean_correction_patterns = self._initialize_korean_corrections()
        
    def _initialize_korean_corrections(self) -> Dict[str, str]:
        """한국어 교정 패턴 초기화"""
        return {
            # 한자 -> 한국어 교정
            r'個人情報': '개인정보',
            r'電子金融': '전자금융',
            r'情報保安': '정보보안',
            r'暗號化': '암호화',
            r'管理體系': '관리체계',
            r'法令': '법령',
            r'規定': '규정',
            r'措置': '조치',
            r'保護': '보호',
            r'安全性': '안전성',
            
            # 영어 -> 한국어 교정
            r'\bsecurity\b': '보안',
            r'\bprivacy\b': '개인정보보호',
            r'\bencryption\b': '암호화',
            r'\bauthentication\b': '인증',
            r'\bauthorization\b': '권한부여',
            r'\bmanagement\b': '관리',
            r'\bsystem\b': '시스템',
            r'\bpolicy\b': '정책',
            r'\bprocedure\b': '절차',
            r'\bincident\b': '사고',
            
            # 잘못된 표현 교정
            r'개인 정보': '개인정보',
            r'전자 금융': '전자금융',
            r'정보 보안': '정보보안',
            r'암호 화': '암호화',
            r'관리 체계': '관리체계',
            r'접근 통제': '접근통제',
            r'위험 평가': '위험평가',
            r'재해 복구': '재해복구',
            r'침해 사고': '침해사고',
            r'보안 정책': '보안정책'
        }
    
    def _evaluate_korean_quality(self, text: str) -> float:
        """한국어 품질 평가"""
        
        if not text:
            return 0.0
        
        # 객관식은 숫자만 확인
        if re.match(r'^[1-5]$', text.strip()):
            return 1.0
        
        # 한자 확인
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        # 한국어 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        
        # 영어 비율
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        # 품질 점수 계산
        quality = korean_ratio * 0.8
        quality -= english_ratio * 0.3
        quality = max(0, min(1, quality))
        
        return quality
    
    def _apply_korean_corrections(self, text: str) -> str:
        """한국어 교정 적용"""
        
        corrected = text
        
        # 교정 패턴 적용
        for pattern, replacement in self.korean_correction_patterns.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # 남은 한자 제거
        corrected = re.sub(r'[\u4e00-\u9fff]+', '', corrected)
        
        # 중복 공백 정리
        corrected = re.sub(r'\s+', ' ', corrected)
        
        return corrected.strip()
    
    def add_correction(self, question: str, predicted: str, 
                      correct: str, reason: str = "",
                      question_id: str = None) -> None:
        """교정 추가 - 한국어 품질 개선 포함"""
        
        if question_id is None:
            question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # 한국어 교정 자동 적용
        corrected_answer = self._apply_korean_corrections(correct)
        
        # 한국어 품질 평가
        original_quality = self._evaluate_korean_quality(predicted)
        corrected_quality = self._evaluate_korean_quality(corrected_answer)
        
        # 신뢰도 부스트 계산
        if predicted == corrected_answer:
            confidence_boost = 0.1
        elif corrected_quality > original_quality:
            confidence_boost = 0.4  # 한국어 개선에 높은 보너스
            self.correction_stats["korean_improvements"] += 1
        else:
            confidence_boost = 0.3
        
        entry = CorrectionEntry(
            question_id=question_id,
            question=question,
            predicted_answer=predicted,
            corrected_answer=corrected_answer,
            correction_reason=reason,
            confidence_boost=confidence_boost,
            korean_quality=corrected_quality
        )
        
        self.corrections.append(entry)
        self.correction_stats["total_corrections"] += 1
        
        # 객관식/주관식 구분
        if corrected_answer.isdigit() and 1 <= int(corrected_answer) <= 5:
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
                self.correction_patterns[keyword][entry.corrected_answer] = {
                    "count": 0,
                    "quality": 0.0
                }
            
            self.correction_patterns[keyword][entry.corrected_answer]["count"] += 1
            self.correction_patterns[keyword][entry.corrected_answer]["quality"] = max(
                self.correction_patterns[keyword][entry.corrected_answer]["quality"],
                entry.korean_quality
            )
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """핵심 구문 추출"""
        
        phrases = []
        
        # 법령 표현
        laws = re.findall(r'\w+법', text.lower())
        phrases.extend(laws)
        
        # 핵심 용어
        key_terms = ["개인정보", "전자금융", "암호화", "보안", "관리체계", "접근통제", "정보보호"]
        for term in key_terms:
            if term in text.lower():
                phrases.append(term)
        
        # 부정형 표현
        if "해당하지" in text or "적절하지" in text:
            phrases.append("부정형")
        
        # 문제 유형
        if "정의" in text:
            phrases.append("정의문제")
        if "방안" in text or "대책" in text:
            phrases.append("방안문제")
        
        return phrases
    
    def apply_corrections(self, question: str, predicted: str) -> Tuple[str, float]:
        """교정 적용 - 한국어 품질 우선"""
        
        # 직접 매칭
        for correction in self.corrections:
            if correction.question == question:
                return correction.corrected_answer, 0.9
        
        # 패턴 기반 교정
        keywords = self._extract_key_phrases(question)
        answer_candidates = {}
        
        for keyword in keywords:
            if keyword in self.correction_patterns:
                for answer, info in self.correction_patterns[keyword].items():
                    if answer not in answer_candidates:
                        answer_candidates[answer] = {
                            "score": 0,
                            "quality": 0
                        }
                    answer_candidates[answer]["score"] += info["count"]
                    answer_candidates[answer]["quality"] = max(
                        answer_candidates[answer]["quality"],
                        info["quality"]
                    )
        
        if answer_candidates:
            # 한국어 품질과 빈도를 모두 고려
            best_answer = max(
                answer_candidates.items(),
                key=lambda x: x[1]["score"] * 0.6 + x[1]["quality"] * 0.4
            )
            confidence = min(
                best_answer[1]["score"] / sum(c["score"] for c in answer_candidates.values()),
                0.8
            )
            
            # 한국어 교정 적용
            corrected = self._apply_korean_corrections(best_answer[0])
            return corrected, confidence
        
        # 기본 한국어 교정만 적용
        corrected = self._apply_korean_corrections(predicted)
        if corrected != predicted:
            self.correction_stats["korean_improvements"] += 1
            return corrected, 0.3
        
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
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['id', 'question', 'predicted', 'correct', 'reason', 'korean_quality']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for correction in self.corrections:
                writer.writerow({
                    'id': correction.question_id,
                    'question': correction.question[:200],
                    'predicted': correction.predicted_answer,
                    'correct': correction.corrected_answer,
                    'reason': correction.correction_reason,
                    'korean_quality': f"{correction.korean_quality:.2f}"
                })
    
    def interactive_correction(self, questions: List[Dict], 
                             predictions: List[str]) -> int:
        """대화형 교정 - 한국어 품질 표시"""
        
        corrections_made = 0
        
        print("\n=== 수동 교정 모드 ===")
        print("각 문제의 예측 답변을 확인하고 교정하세요.")
        print("Enter: 수락, 숫자/텍스트: 교정, q: 종료\n")
        
        for i, (q_data, pred) in enumerate(zip(questions, predictions)):
            print(f"\n[{i+1}/{len(questions)}]")
            print(f"문제: {q_data['question'][:150]}...")
            print(f"예측 답변: {pred}")
            
            # 한국어 품질 표시
            quality = self._evaluate_korean_quality(pred)
            if quality < 0.5:
                print(f"⚠️ 한국어 품질 낮음 ({quality:.2f})")
            
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
                
                # 한국어 품질 개선 확인
                new_quality = self._evaluate_korean_quality(user_input)
                if new_quality > quality:
                    print(f"✓ 한국어 품질 개선 ({quality:.2f} -> {new_quality:.2f})")
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
            "korean_improvements": self.correction_stats["korean_improvements"],
            "patterns": len(self.correction_patterns),
            "pattern_distribution": {
                pattern: len(answers) 
                for pattern, answers in self.correction_patterns.items()
            },
            "avg_korean_quality": sum(c.korean_quality for c in self.corrections) / max(len(self.corrections), 1)
        }
    
    def generate_korean_quality_report(self) -> Dict:
        """한국어 품질 보고서"""
        
        if not self.corrections:
            return {"status": "교정 데이터 없음"}
        
        qualities = [c.korean_quality for c in self.corrections]
        
        return {
            "total_corrections": len(self.corrections),
            "avg_quality": sum(qualities) / len(qualities),
            "high_quality": sum(1 for q in qualities if q > 0.8),
            "medium_quality": sum(1 for q in qualities if 0.5 < q <= 0.8),
            "low_quality": sum(1 for q in qualities if q <= 0.5),
            "korean_improvements": self.correction_stats["korean_improvements"],
            "improvement_rate": self.correction_stats["korean_improvements"] / len(self.corrections)
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.corrections:
            self.save_corrections_to_csv()
        print(f"교정 시스템: {len(self.corrections)}개 교정 저장")
        if self.correction_stats["korean_improvements"] > 0:
            print(f"한국어 품질 개선: {self.correction_stats['korean_improvements']}회")