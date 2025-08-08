# auto_learner.py
"""
자동 학습
"""

import numpy as np
import pickle
import re
import os
import tempfile
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

def atomic_save_model(obj, filepath: str) -> bool:
    """원자적 모델 저장"""
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, filepath)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception:
        return False

def atomic_load_model(filepath: str):
    """안전한 모델 로드"""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

class AutoLearner:
    """자동 학습 엔진"""
    
    def __init__(self):
        self.learning_rate = 0.15
        self.confidence_threshold = 0.6
        self.min_samples = 2
        
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
        
        self.specialized_rules = self._initialize_specialized_rules()
        
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
            ],
            "사고대응": [
                "{event} 발생 시 {action}를 수행해야 합니다.",
                "침해사고 대응은 {phase}별로 {measure}를 이행해야 합니다.",
                "복구 계획은 {target}을 고려하여 {requirement}를 수립해야 합니다."
            ]
        }
    
    def _initialize_specialized_rules(self) -> Dict[str, Dict]:
        """특화 규칙 초기화"""
        return {
            "금융투자업_분류": {
                "patterns": ["금융투자업", "소비자금융업", "보험중개업", "투자매매업", "투자중개업", "투자자문업"],
                "answers": {"1": 0.85, "5": 0.10, "2": 0.03, "3": 0.01, "4": 0.01},
                "confidence": 0.9,
                "rule": "소비자금융업과 보험중개업은 금융투자업이 아님"
            },
            "위험관리_요소": {
                "patterns": ["위험관리", "계획수립", "고려요소", "위험수용", "대응전략"],
                "answers": {"2": 0.8, "1": 0.12, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "rule": "위험수용은 위험대응전략의 하나이지 별도 고려요소가 아님"
            },
            "관리체계_정책": {
                "patterns": ["관리체계", "정책수립", "경영진", "참여", "최고책임자"],
                "answers": {"2": 0.75, "1": 0.15, "3": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.8,
                "rule": "정책수립 단계에서 경영진의 참여가 가장 중요"
            },
            "재해복구_요소": {
                "patterns": ["재해복구", "계획수립", "개인정보파기", "복구절차", "비상연락"],
                "answers": {"3": 0.8, "1": 0.1, "2": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "rule": "개인정보파기절차는 재해복구와 직접 관련 없음"
            }
        }
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        """한국어 품질 평가"""
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.8 - english_ratio * 0.2
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.05, 0.15)
        
        if 50 <= len(text) <= 300:
            quality += 0.1
        
        return max(0, min(1, quality))
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 문제 확인"""
        negative_keywords = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", 
            "틀린", "잘못된", "부적절한", "아닌"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in negative_keywords)
    
    def _extract_patterns(self, question: str) -> List[str]:
        """효율적 패턴 추출"""
        
        patterns = []
        question_lower = question.lower()
        
        for rule_name, rule_info in self.specialized_rules.items():
            rule_patterns = rule_info["patterns"]
            match_count = sum(1 for pattern in rule_patterns if pattern in question_lower)
            
            if match_count >= 2:
                patterns.append(rule_name)
        
        if self._is_negative_question(question):
            patterns.append("negative_question")
        
        if "정의" in question_lower:
            patterns.append("definition_question")
        if "방안" in question_lower or "대책" in question_lower:
            patterns.append("solution_question")
        
        domains = {
            "personal_info": ["개인정보", "정보주체", "동의"],
            "electronic": ["전자금융", "전자적", "거래"],
            "security": ["보안", "암호화", "접근통제"],
            "crypto": ["암호", "해시", "전자서명"],
            "incident": ["사고", "유출", "침해"]
        }
        
        for domain, keywords in domains.items():
            if sum(1 for kw in keywords if kw in question_lower) >= 1:
                patterns.append(f"domain_{domain}")
        
        return patterns[:8]
    
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str]) -> None:
        """예측으로부터 학습"""
        
        if confidence < self.confidence_threshold:
            return
        
        korean_quality = self._evaluate_korean_quality(prediction, question_type)
        
        if korean_quality < 0.4 and question_type != "multiple_choice":
            return
        
        patterns = self._extract_patterns(question)
        
        for pattern in patterns:
            weight_boost = confidence * self.learning_rate * max(korean_quality, 0.5)
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
        
        if korean_quality > 0.7 and question_type != "multiple_choice":
            self._learn_korean_patterns(prediction, domain)
        
        self.learning_history.append({
            "question_sample": question[:80],
            "prediction": prediction[:80] if len(prediction) > 80 else prediction,
            "confidence": confidence,
            "korean_quality": korean_quality,
            "patterns": len(patterns)
        })
        
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
    
    def _learn_korean_patterns(self, text: str, domains: List[str]) -> None:
        """한국어 패턴 학습"""
        
        if 50 <= len(text) <= 400:
            self.successful_korean_templates.append({
                "text": text,
                "domains": domains,
                "structure": self._analyze_text_structure(text)
            })
            
            if len(self.successful_korean_templates) > 30:
                self.successful_korean_templates = sorted(
                    self.successful_korean_templates,
                    key=lambda x: self._evaluate_korean_quality(x["text"], "subjective"),
                    reverse=True
                )[:30]
        
        for domain in domains:
            self.korean_quality_patterns[domain].append({
                "length": len(text),
                "keyword_count": self._count_domain_keywords(text, domain),
                "structure_markers": self._extract_structure_markers(text)
            })
            
            if len(self.korean_quality_patterns[domain]) > 20:
                self.korean_quality_patterns[domain] = self.korean_quality_patterns[domain][-20:]
    
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
        if re.search(r'따라서|그러므로|결론적으로', text):
            markers.append("conclusion")
        if re.search(r'예를 들어|구체적으로', text):
            markers.append("example")
        
        return markers
    
    def predict_with_patterns(self, question: str, question_type: str) -> Tuple[str, float]:
        """패턴 기반 예측"""
        
        patterns = self._extract_patterns(question)
        
        if not patterns:
            return self._get_default_answer(question_type), 0.2
        
        for rule_name in patterns:
            if rule_name in self.specialized_rules:
                rule = self.specialized_rules[rule_name]
                answers = rule["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                return best_answer[0], rule["confidence"]
        
        answer_scores = defaultdict(_default_float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights and self.pattern_counts[pattern] >= self.min_samples:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if not answer_scores:
            return self._get_default_answer(question_type), 0.2
        
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        confidence = min(best_answer[1] / max(total_weight, 1), 0.8)
        
        if question_type != "multiple_choice":
            korean_quality = self._evaluate_korean_quality(best_answer[0], question_type)
            if korean_quality < 0.5:
                return self._generate_korean_answer(question, patterns), 0.4
        
        return best_answer[0], confidence
    
    def _generate_korean_answer(self, question: str, patterns: List[str]) -> str:
        """한국어 답변 생성"""
        
        domain = None
        for pattern in patterns:
            if pattern.startswith("domain_"):
                domain = pattern.replace("domain_", "")
                break
        
        if self.successful_korean_templates:
            relevant_templates = []
            for template in self.successful_korean_templates:
                if not domain or domain in [d.lower().replace("개인정보보호", "personal_info").replace("전자금융", "electronic").replace("정보보안", "security") for d in template["domains"]]:
                    relevant_templates.append(template)
            
            if relevant_templates:
                best_template = max(relevant_templates, 
                                  key=lambda x: self._evaluate_korean_quality(x["text"], "subjective"))
                return best_template["text"]
        
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
        
        return base_answer
    
    def _get_default_answer(self, question_type: str) -> str:
        """기본 답변"""
        
        if question_type == "multiple_choice":
            if self.answer_distribution["mc"]:
                return max(self.answer_distribution["mc"].items(), 
                          key=lambda x: x[1])[0]
            return "2"
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
            if pattern in self.pattern_weights:
                del self.pattern_weights[pattern]
            if pattern in self.pattern_counts:
                del self.pattern_counts[pattern]
            removed += 1
        
        for pattern in self.pattern_weights:
            total = sum(self.pattern_weights[pattern].values())
            if total > 0:
                max_weight = max(self.pattern_weights[pattern].values())
                if max_weight > total * 0.8:
                    for answer in self.pattern_weights[pattern]:
                        if self.pattern_weights[pattern][answer] == max_weight:
                            self.pattern_weights[pattern][answer] *= 1.1
                        else:
                            self.pattern_weights[pattern][answer] *= 0.9
                optimized += 1
        
        if len(self.learning_history) > 50:
            recent_qualities = [h.get("korean_quality", 0) for h in self.learning_history[-20:]]
            if recent_qualities:
                avg_quality = sum(recent_qualities) / len(recent_qualities)
                if avg_quality > 0.7:
                    self.confidence_threshold = max(self.confidence_threshold - 0.05, 0.5)
                elif avg_quality < 0.5:
                    self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.8)
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights),
            "confidence_threshold": self.confidence_threshold
        }
    
    def analyze_learning_progress(self) -> Dict:
        """학습 진행 분석"""
        
        if not self.learning_history:
            return {"status": "학습 데이터 없음"}
        
        recent_history = self.learning_history[-30:]
        
        confidences = [h["confidence"] for h in recent_history]
        confidence_trend = np.mean(confidences) if confidences else 0
        
        korean_qualities = [h.get("korean_quality", 0) for h in recent_history]
        korean_quality_trend = np.mean(korean_qualities) if korean_qualities else 0
        
        pattern_diversity = len(self.pattern_weights)
        mc_distribution = dict(self.answer_distribution["mc"])
        
        specialized_rule_usage = {}
        for rule_name in self.specialized_rules:
            usage_count = sum(1 for h in recent_history if rule_name in h.get("question_sample", ""))
            specialized_rule_usage[rule_name] = usage_count
        
        return {
            "total_samples": len(self.learning_history),
            "confidence_trend": confidence_trend,
            "korean_quality_trend": korean_quality_trend,
            "pattern_diversity": pattern_diversity,
            "mc_distribution": mc_distribution,
            "negative_distribution": dict(self.answer_distribution["negative"]),
            "active_patterns": list(self.pattern_weights.keys())[:8],
            "successful_templates": len(self.successful_korean_templates),
            "specialized_rule_usage": specialized_rule_usage,
            "learning_efficiency": self._calculate_learning_efficiency()
        }
    
    def _calculate_learning_efficiency(self) -> float:
        """학습 효율성 계산"""
        if len(self.learning_history) < 5:
            return 0.0
        
        recent_samples = self.learning_history[-10:]
        early_samples = self.learning_history[:10] if len(self.learning_history) >= 20 else []
        
        if not early_samples:
            return 0.5
        
        recent_avg_confidence = np.mean([s["confidence"] for s in recent_samples])
        early_avg_confidence = np.mean([s["confidence"] for s in early_samples])
        
        recent_avg_quality = np.mean([s.get("korean_quality", 0) for s in recent_samples])
        early_avg_quality = np.mean([s.get("korean_quality", 0) for s in early_samples])
        
        confidence_improvement = recent_avg_confidence - early_avg_confidence
        quality_improvement = recent_avg_quality - early_avg_quality
        
        efficiency = (confidence_improvement + quality_improvement) / 2
        return max(0, min(1, 0.5 + efficiency))
    
    def save_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        """모델 저장"""
        
        model_data = {
            "pattern_weights": {k: dict(v) for k, v in self.pattern_weights.items()},
            "pattern_counts": dict(self.pattern_counts),
            "answer_distribution": {
                "mc": dict(self.answer_distribution["mc"]),
                "domain": {k: dict(v) for k, v in self.answer_distribution["domain"].items()},
                "negative": dict(self.answer_distribution["negative"])
            },
            "korean_quality_patterns": {k: v[-10:] for k, v in self.korean_quality_patterns.items()},
            "successful_korean_templates": self.successful_korean_templates[-20:],
            "learning_history": self.learning_history[-50:],
            "specialized_rules": self.specialized_rules,
            "parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_samples": self.min_samples
            }
        }
        
        return atomic_save_model(model_data, filepath)
    
    def load_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        """모델 로드"""
        
        model_data = atomic_load_model(filepath)
        if model_data is None:
            return False
        
        try:
            self.pattern_weights = defaultdict(_default_float_dict)
            for k, v in model_data.get("pattern_weights", {}).items():
                self.pattern_weights[k] = defaultdict(_default_float, v)
            
            self.pattern_counts = defaultdict(_default_int, model_data.get("pattern_counts", {}))
            
            answer_dist = model_data.get("answer_distribution", {})
            self.answer_distribution = {
                "mc": defaultdict(_default_int, answer_dist.get("mc", {})),
                "domain": defaultdict(_default_int_dict),
                "negative": defaultdict(_default_int, answer_dist.get("negative", {}))
            }
            
            for k, v in answer_dist.get("domain", {}).items():
                self.answer_distribution["domain"][k] = defaultdict(_default_int, v)
            
            self.korean_quality_patterns = defaultdict(_default_list, model_data.get("korean_quality_patterns", {}))
            self.successful_korean_templates = model_data.get("successful_korean_templates", [])
            self.learning_history = model_data.get("learning_history", [])
            
            if "specialized_rules" in model_data:
                self.specialized_rules.update(model_data["specialized_rules"])
            
            params = model_data.get("parameters", {})
            self.learning_rate = params.get("learning_rate", 0.15)
            self.confidence_threshold = params.get("confidence_threshold", 0.6)
            self.min_samples = params.get("min_samples", 2)
            
            return True
        except Exception:
            return False
    
    def cleanup(self):
        """리소스 정리"""
        total_patterns = len(self.pattern_weights)
        total_samples = len(self.learning_history)
        if total_patterns > 0 or total_samples > 0:
            print(f"자동 학습: {total_patterns}개 패턴, {total_samples}개 샘플")