# auto_learner.py
"""
자동 학습
"""

import numpy as np
import pickle
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def _default_int():
    """기본 정수값 반환"""
    return 0

def _default_float():
    """기본 실수값 반환"""
    return 0.0

def _default_list():
    """기본 리스트 반환"""
    return []

def _default_float_dict():
    """기본 실수 딕셔너리 반환"""
    return defaultdict(_default_float)

def _default_int_dict():
    """기본 정수 딕셔너리 반환"""
    return defaultdict(_default_int)

class AutoLearner:
    """자동 학습 엔진"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_samples = 5
        
        self.pattern_weights = defaultdict(_default_float_dict)
        self.pattern_counts = defaultdict(_default_int)
        
        self.answer_distribution = {
            "mc": defaultdict(_default_int),
            "domain": defaultdict(_default_int_dict),
            "negative": defaultdict(_default_int)
        }
        
        self.korean_quality_patterns = defaultdict(_default_list)
        self.successful_korean_templates = []
        
        self.learning_history = []
        self.performance_metrics = []
        
        self.domain_templates = self._initialize_korean_templates()
        
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        """도메인별 한국어 템플릿 초기화"""
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 {action}가 필요합니다.",
                "정보주체의 권리 보호를 위해 {measure}를 수행해야 합니다.",
                "개인정보의 안전한 관리를 위해 {requirement}가 요구됩니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {action}를 수행해야 합니다.",
                "전자적 장치를 통한 거래의 안전성 확보를 위해 {measure}가 필요합니다.",
                "접근매체 관리와 관련하여 {requirement}를 준수해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계에 따라 {action}를 구현해야 합니다.",
                "체계적인 보안 관리를 위해 {measure}가 요구됩니다.",
                "위험평가를 통해 {requirement}를 수립해야 합니다."
            ],
            "암호화": [
                "중요 정보는 {action}를 통해 보호해야 합니다.",
                "암호화 기술을 활용하여 {measure}를 확보해야 합니다.",
                "안전한 키 관리를 위해 {requirement}가 필요합니다."
            ]
        }
    
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str]) -> None:
        """예측으로부터 학습"""
        
        if confidence < self.confidence_threshold:
            return
        
        korean_quality = self._evaluate_korean_quality(prediction, question_type)
        
        if korean_quality < 0.5 and question_type != "multiple_choice":
            return
        
        patterns = self._extract_patterns(question)
        
        for pattern in patterns:
            weight_boost = confidence * self.learning_rate * korean_quality
            self.pattern_weights[pattern][prediction] += weight_boost
            self.pattern_counts[pattern] += 1
        
        if question_type == "multiple_choice":
            self.answer_distribution["mc"][prediction] += 1
            
            if self._is_negative_question(question):
                self.answer_distribution["negative"][prediction] += 1
        
        for d in domain:
            if d not in self.answer_distribution["domain"]:
                self.answer_distribution["domain"][d] = defaultdict(_default_int)
            self.answer_distribution["domain"][d][prediction] += 1
        
        if korean_quality > 0.8 and question_type != "multiple_choice":
            self._learn_korean_patterns(prediction, domain)
        
        self.learning_history.append({
            "question_sample": question[:100],
            "prediction": prediction[:100] if len(prediction) > 100 else prediction,
            "confidence": confidence,
            "korean_quality": korean_quality,
            "patterns": len(patterns)
        })
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        """한국어 품질 평가"""
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.8
        quality -= english_ratio * 0.4
        quality = max(0, min(1, quality))
        
        return quality
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 문제 확인"""
        negative_keywords = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", 
            "틀린", "잘못된", "부적절한", "아닌"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in negative_keywords)
    
    def _extract_patterns(self, question: str) -> List[str]:
        """패턴 추출"""
        
        patterns = []
        question_lower = question.lower()
        
        if self._is_negative_question(question):
            patterns.append("negative_question")
        if "정의" in question_lower:
            patterns.append("definition_question")
        if "법" in question_lower and ("따르면" in question_lower or "의하면" in question_lower):
            patterns.append("law_based")
        if "방안" in question_lower or "대책" in question_lower:
            patterns.append("solution_question")
        if "절차" in question_lower or "과정" in question_lower:
            patterns.append("procedure_question")
        
        domains = {
            "personal_info": ["개인정보", "정보주체", "동의", "수집", "이용"],
            "electronic": ["전자금융", "전자적", "거래", "접근매체"],
            "security": ["보안", "암호화", "접근통제", "관리체계"],
            "crypto": ["암호", "해시", "전자서명", "인증서"],
            "incident": ["사고", "유출", "침해", "대응", "복구"]
        }
        
        for domain, keywords in domains.items():
            if any(kw in question_lower for kw in keywords):
                patterns.append(f"domain_{domain}")
        
        if len(question) < 200:
            patterns.append("short_question")
        elif len(question) > 500:
            patterns.append("long_question")
        else:
            patterns.append("medium_question")
        
        if question.count('\n') > 5:
            patterns.append("complex_structure")
        if len(re.findall(r'[1-5]', question)) >= 5:
            patterns.append("has_choices")
        
        return patterns
    
    def _learn_korean_patterns(self, text: str, domains: List[str]) -> None:
        """한국어 패턴 학습"""
        
        if len(text) > 50 and len(text) < 800:
            self.successful_korean_templates.append({
                "text": text,
                "domains": domains,
                "structure": self._analyze_text_structure(text)
            })
            
            if len(self.successful_korean_templates) > 100:
                self.successful_korean_templates = self.successful_korean_templates[-100:]
        
        for domain in domains:
            self.korean_quality_patterns[domain].append({
                "length": len(text),
                "keyword_count": self._count_domain_keywords(text, domain),
                "structure_markers": self._extract_structure_markers(text)
            })
    
    def _analyze_text_structure(self, text: str) -> Dict:
        """텍스트 구조 분석"""
        return {
            "has_numbering": bool(re.search(r'첫째|둘째|1\)|2\)', text)),
            "has_law_reference": bool(re.search(r'법|규정|조항', text)),
            "has_conclusion": bool(re.search(r'따라서|그러므로|결론적으로', text)),
            "sentence_count": len(re.split(r'[.!?]', text))
        }
    
    def _count_domain_keywords(self, text: str, domain: str) -> int:
        """도메인 키워드 수 계산"""
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "동의", "수집", "이용", "제공", "파기"],
            "전자금융": ["전자금융", "전자적", "거래", "접근매체", "전자서명"],
            "정보보안": ["보안", "관리체계", "접근통제", "위험평가", "보호대책"],
            "암호화": ["암호", "암호화", "복호화", "키", "해시", "전자서명"]
        }
        
        keywords = domain_keywords.get(domain, [])
        count = sum(1 for kw in keywords if kw in text)
        return count
    
    def _extract_structure_markers(self, text: str) -> List[str]:
        """구조 마커 추출"""
        markers = []
        
        if re.search(r'첫째|둘째|셋째', text):
            markers.append("ordered_list")
        if re.search(r'1\)|2\)|3\)', text):
            markers.append("numbered_list")
        if re.search(r'가\.|나\.|다\.', text):
            markers.append("korean_list")
        if re.search(r'따라서|그러므로|결론적으로', text):
            markers.append("conclusion")
        if re.search(r'예를 들어|구체적으로', text):
            markers.append("example")
        
        return markers
    
    def predict_with_patterns(self, question: str, question_type: str) -> Tuple[str, float]:
        """패턴 기반 예측"""
        
        patterns = self._extract_patterns(question)
        
        if not patterns:
            return self._get_default_answer(question_type), 0.3
        
        answer_scores = defaultdict(_default_float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if not answer_scores:
            return self._get_default_answer(question_type), 0.3
        
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        confidence = min(best_answer[1] / max(total_weight, 1), 0.9)
        
        if question_type != "multiple_choice":
            korean_quality = self._evaluate_korean_quality(best_answer[0], question_type)
            if korean_quality < 0.5:
                return self._generate_korean_answer(question, patterns), 0.5
        
        return best_answer[0], confidence
    
    def _generate_korean_answer(self, question: str, patterns: List[str]) -> str:
        """한국어 답변 생성"""
        
        domain = None
        for pattern in patterns:
            if pattern.startswith("domain_"):
                domain = pattern.replace("domain_", "")
                break
        
        if domain == "personal_info":
            base_answer = "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif domain == "electronic":
            base_answer = "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다."
        elif domain == "security":
            base_answer = "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다."
        elif domain == "crypto":
            base_answer = "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화하고 안전한 키 관리 체계를 구축해야 합니다."
        else:
            base_answer = "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선을 수행해야 합니다."
        
        if "solution_question" in patterns:
            base_answer += " 구체적인 방안으로는 정책 수립, 조직 구성, 기술적 대책 구현, 정기적 점검 등이 있습니다."
        elif "procedure_question" in patterns:
            base_answer += " 절차는 계획 수립, 실행, 모니터링, 개선의 단계로 체계적으로 수행되어야 합니다."
        
        return base_answer
    
    def _get_default_answer(self, question_type: str) -> str:
        """기본 답변"""
        
        if question_type == "multiple_choice":
            if self.answer_distribution["mc"]:
                return max(self.answer_distribution["mc"].items(), 
                          key=lambda x: x[1])[0]
            return "3"
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 통해 안전성을 확보해야 합니다."
    
    def optimize_patterns(self) -> Dict:
        """패턴 최적화"""
        
        optimized = 0
        removed = 0
        
        patterns_to_remove = []
        for pattern, count in self.pattern_counts.items():
            if count < self.min_samples:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            del self.pattern_weights[pattern]
            del self.pattern_counts[pattern]
            removed += 1
        
        for pattern in self.pattern_weights:
            total = sum(self.pattern_weights[pattern].values())
            if total > 0:
                for answer in self.pattern_weights[pattern]:
                    self.pattern_weights[pattern][answer] /= total
                optimized += 1
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights)
        }
    
    def analyze_learning_progress(self) -> Dict:
        """학습 진행 분석"""
        
        if not self.learning_history:
            return {"status": "학습 데이터 없음"}
        
        recent_history = self.learning_history[-100:]
        
        confidences = [h["confidence"] for h in recent_history]
        confidence_trend = np.mean(confidences) if confidences else 0
        
        korean_qualities = [h.get("korean_quality", 0) for h in recent_history]
        korean_quality_trend = np.mean(korean_qualities) if korean_qualities else 0
        
        pattern_diversity = len(self.pattern_weights)
        
        mc_distribution = dict(self.answer_distribution["mc"])
        
        return {
            "total_samples": len(self.learning_history),
            "confidence_trend": confidence_trend,
            "korean_quality_trend": korean_quality_trend,
            "pattern_diversity": pattern_diversity,
            "mc_distribution": mc_distribution,
            "negative_distribution": dict(self.answer_distribution["negative"]),
            "active_patterns": list(self.pattern_weights.keys())[:10],
            "successful_templates": len(self.successful_korean_templates)
        }
    
    def save_model(self, filepath: str = "./auto_learner_model.pkl") -> None:
        """모델 저장"""
        
        model_data = {
            "pattern_weights": dict(self.pattern_weights),
            "pattern_counts": dict(self.pattern_counts),
            "answer_distribution": {
                "mc": dict(self.answer_distribution["mc"]),
                "domain": {k: dict(v) for k, v in self.answer_distribution["domain"].items()},
                "negative": dict(self.answer_distribution["negative"])
            },
            "korean_quality_patterns": dict(self.korean_quality_patterns),
            "successful_korean_templates": self.successful_korean_templates[-50:],
            "learning_history": self.learning_history[-1000:],
            "parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_samples": self.min_samples
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        """모델 로드"""
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pattern_weights = defaultdict(_default_float_dict)
            for k, v in model_data["pattern_weights"].items():
                self.pattern_weights[k] = defaultdict(_default_float, v)
            
            self.pattern_counts = defaultdict(_default_int, model_data["pattern_counts"])
            
            answer_dist = model_data["answer_distribution"]
            self.answer_distribution = {
                "mc": defaultdict(_default_int, answer_dist.get("mc", {})),
                "domain": defaultdict(_default_int_dict),
                "negative": defaultdict(_default_int, answer_dist.get("negative", {}))
            }
            
            for k, v in answer_dist.get("domain", {}).items():
                self.answer_distribution["domain"][k] = defaultdict(_default_int, v)
            
            self.korean_quality_patterns = defaultdict(_default_list, model_data.get("korean_quality_patterns", {}))
            self.successful_korean_templates = model_data.get("successful_korean_templates", [])
            self.learning_history = model_data["learning_history"]
            
            params = model_data.get("parameters", {})
            self.learning_rate = params.get("learning_rate", 0.1)
            self.confidence_threshold = params.get("confidence_threshold", 0.7)
            self.min_samples = params.get("min_samples", 5)
            
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        print(f"자동 학습: {len(self.pattern_weights)}개 패턴 학습")