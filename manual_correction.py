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
    """수동 교정 시스템"""
    
    def __init__(self):
        self.corrections = []
        self.correction_patterns = {}
        self.correction_stats = {
            "total_corrections": 0,
            "mc_corrections": 0,
            "subj_corrections": 0,
            "korean_improvements": 0
        }
        
        # 핵심 한국어 교정만 유지
        self.korean_correction_patterns = self._initialize_korean_corrections()
        
        # 간소화된 캐시
        self.correction_cache = {}
        self.max_cache_size = 100
        
    def _initialize_korean_corrections(self) -> Dict[str, str]:
        """핵심 한국어 교정 패턴"""
        return {
            # 주요 한자 -> 한국어
            r'個人情報|个人信息': '개인정보',
            r'電子金融|电子金融': '전자금융',
            r'情報保安|信息安全': '정보보안',
            r'暗號化|加密': '암호화',
            r'管理體系|管理体系': '관리체계',
            
            # 주요 영어 -> 한국어
            r'\bsecurity\b': '보안',
            r'\bprivacy\b': '개인정보보호',
            r'\bencryption\b': '암호화',
            r'\bauthentication\b': '인증',
            r'\bmanagement\b': '관리',
            r'\bsystem\b': '시스템',
            
            # 띄어쓰기 교정
            r'개인 정보': '개인정보',
            r'전자 금융': '전자금융',
            r'정보 보안': '정보보안',
            r'암호 화': '암호화',
            r'관리 체계': '관리체계'
        }
    
    def _evaluate_korean_quality(self, text: str) -> float:
        """간소화된 한국어 품질 평가"""
        
        if not text:
            return 0.0
        
        # 객관식 확인
        if re.match(r'^[1-5]$', text.strip()):
            return 1.0
        
        # 한자 확인
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        # 한국어 비율
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_ratio = len(re.findall(r'[A-Za-z]', text)) / total_chars
        
        # 간단한 품질 점수
        quality = korean_ratio * 0.8 - english_ratio * 0.3
        return max(0, min(1, quality))
    
    def _apply_korean_corrections(self, text: str) -> str:
        """한국어 교정 적용"""
        
        corrected = text
        
        # 교정 패턴 적용
        for pattern, replacement in self.korean_correction_patterns.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # 한자 제거
        corrected = re.sub(r'[\u4e00-\u9fff]+', '', corrected)
        
        # 공백 정리
        corrected = re.sub(r'\s+', ' ', corrected)
        
        return corrected.strip()
    
    def add_correction(self, question: str, predicted: str, 
                      correct: str, reason: str = "",
                      question_id: str = None) -> None:
        """교정 추가"""
        
        if question_id is None:
            question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # 한국어 교정 적용
        corrected_answer = self._apply_korean_corrections(correct)
        
        # 품질 평가
        original_quality = self._evaluate_korean_quality(predicted)
        corrected_quality = self._evaluate_korean_quality(corrected_answer)
        
        # 신뢰도 계산
        if predicted == corrected_answer:
            confidence_boost = 0.1
        elif corrected_quality > original_quality:
            confidence_boost = 0.4
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
        """간소화된 패턴 학습"""
        
        # 핵심 키워드만 추출
        keywords = self._extract_key_phrases_simple(entry.question)
        
        for keyword in keywords[:3]:  # 최대 3개만
            if keyword not in self.correction_patterns:
                self.correction_patterns[keyword] = {}
            
            answer = entry.corrected_answer
            if answer not in self.correction_patterns[keyword]:
                self.correction_patterns[keyword][answer] = {"count": 0, "quality": 0.0}
            
            self.correction_patterns[keyword][answer]["count"] += 1
            self.correction_patterns[keyword][answer]["quality"] = max(
                self.correction_patterns[keyword][answer]["quality"],
                entry.korean_quality
            )
    
    def _extract_key_phrases_simple(self, text: str) -> List[str]:
        """간소화된 키워드 추출"""
        
        phrases = []
        text_lower = text.lower()
        
        # 핵심 용어만
        key_terms = ["개인정보", "전자금융", "암호화", "보안", "관리체계"]
        for term in key_terms:
            if term in text_lower:
                phrases.append(term)
        
        # 문제 유형
        if "해당하지" in text_lower or "적절하지" in text_lower:
            phrases.append("부정형")
        if "정의" in text_lower:
            phrases.append("정의문제")
        
        return phrases
    
    def apply_corrections(self, question: str, predicted: str) -> Tuple[str, float]:
        """교정 적용"""
        
        # 캐시 확인
        cache_key = hash(f"{question[:50]}{predicted}")
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
        
        # 직접 매칭
        for correction in self.corrections:
            if correction.question == question:
                result = (correction.corrected_answer, 0.9)
                self._cache_result(cache_key, result)
                return result
        
        # 패턴 기반 교정
        keywords = self._extract_key_phrases_simple(question)
        answer_candidates = {}
        
        for keyword in keywords:
            if keyword in self.correction_patterns:
                for answer, info in self.correction_patterns[keyword].items():
                    if answer not in answer_candidates:
                        answer_candidates[answer] = {"score": 0, "quality": 0}
                    answer_candidates[answer]["score"] += info["count"]
                    answer_candidates[answer]["quality"] = max(
                        answer_candidates[answer]["quality"], info["quality"]
                    )
        
        if answer_candidates:
            # 최고 점수 답변 선택
            best_answer = max(
                answer_candidates.items(),
                key=lambda x: x[1]["score"] * 0.7 + x[1]["quality"] * 0.3
            )
            confidence = min(best_answer[1]["score"] / 10, 0.8)
            
            corrected = self._apply_korean_corrections(best_answer[0])
            result = (corrected, confidence)
            self._cache_result(cache_key, result)
            return result
        
        # 기본 한국어 교정
        corrected = self._apply_korean_corrections(predicted)
        if corrected != predicted:
            self.correction_stats["korean_improvements"] += 1
            result = (corrected, 0.3)
        else:
            result = (predicted, 0.0)
        
        self._cache_result(cache_key, result)
        return result
    
    def _cache_result(self, cache_key: int, result: Tuple[str, float]) -> None:
        """캐시 결과 저장"""
        if len(self.correction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.correction_cache))
            del self.correction_cache[oldest_key]
        
        self.correction_cache[cache_key] = result
    
    def load_corrections_from_csv(self, filepath: str) -> int:
        """CSV 로드"""
        
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
        except Exception:
            pass
        
        return loaded
    
    def save_corrections_to_csv(self, filepath: str = "./corrections.csv") -> None:
        """CSV 저장"""
        
        try:
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
        except Exception:
            pass
    
    def interactive_correction(self, questions: List[Dict], 
                             predictions: List[str]) -> int:
        """대화형 교정"""
        
        corrections_made = 0
        
        print("\n수동 교정 모드")
        print("각 문제의 예측 답변을 확인하고 교정하세요.")
        print("Enter: 수락, 숫자/텍스트: 교정, q: 종료\n")
        
        for i, (q_data, pred) in enumerate(zip(questions, predictions)):
            print(f"\n[{i+1}/{len(questions)}]")
            print(f"문제: {q_data['question'][:150]}...")
            print(f"예측 답변: {pred}")
            
            # 한국어 품질 표시
            quality = self._evaluate_korean_quality(pred)
            if quality < 0.5:
                print(f"한국어 품질 낮음 ({quality:.2f})")
            
            user_input = input("교정 (Enter=수락, q=종료): ").strip()
            
            if user_input.lower() == 'q':
                break
            elif user_input:
                reason = input("교정 이유 (선택): ").strip()
                
                self.add_correction(
                    question=q_data['question'],
                    predicted=pred,
                    correct=user_input,
                    reason=reason,
                    question_id=q_data.get('id')
                )
                corrections_made += 1
                print("교정 저장됨")
        
        return corrections_made
    
    def batch_correction(self, correction_file: str) -> int:
        """배치 교정"""
        
        if not os.path.exists(correction_file):
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
        
        except Exception:
            pass
        
        return corrections_made
    
    def get_correction_stats(self) -> Dict:
        """교정 통계"""
        
        return {
            "total": self.correction_stats["total_corrections"],
            "mc": self.correction_stats["mc_corrections"],
            "subjective": self.correction_stats["subj_corrections"],
            "korean_improvements": self.correction_stats["korean_improvements"],
            "patterns": len(self.correction_patterns),
            "cache_size": len(self.correction_cache)
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
        
        self.correction_cache.clear()