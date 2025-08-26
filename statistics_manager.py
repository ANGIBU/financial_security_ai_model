# statistics_manager.py

import time
import sys
from typing import Dict, List
from datetime import datetime
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil을 사용할 수 없습니다. 메모리 모니터링이 제한됩니다.")

from config import LOG_DIR


class StatisticsManager:
    """실행 통계 관리"""
    
    def __init__(self, log_type: str = "inference"):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            if log_type == "test":
                self.log_file = LOG_DIR / "test.txt"
            else:
                self.log_file = LOG_DIR / "inference.txt"
        except Exception as e:
            print(f"로그 디렉토리 생성 실패: {e}")
            self.log_file = Path("inference.txt")
        
        self.start_time = None
        self.processing_times = []
        self.domain_stats = {}
        self.method_stats = {}
        self.error_stats = {}
        self.template_usage_stats = {}
        self.memory_snapshots = []
        self.memory_monitor_enabled = PSUTIL_AVAILABLE
        self.session_started = False
        
    def start_session(self):
        """세션 시작"""
        self.start_time = time.time()
        self.session_started = True
        self._clear_log_and_write_system_info()
        
    def record_question_processing(self, processing_time: float, domain: str, method: str, 
                                  question_type: str, success: bool, error_type: str = None):
        """문항 처리 기록"""
        try:
            self.processing_times.append(processing_time)
            
            # 도메인 통계
            if domain not in self.domain_stats:
                self.domain_stats[domain] = {"count": 0, "avg_time": 0, "total_time": 0, "success_count": 0}
            self.domain_stats[domain]["count"] += 1
            self.domain_stats[domain]["total_time"] += processing_time
            self.domain_stats[domain]["avg_time"] = self.domain_stats[domain]["total_time"] / self.domain_stats[domain]["count"]
            
            if success:
                self.domain_stats[domain]["success_count"] += 1
            
            # 처리 방법 통계
            if method not in self.method_stats:
                self.method_stats[method] = {"count": 0, "success_count": 0, "avg_time": 0, "total_time": 0}
            self.method_stats[method]["count"] += 1
            self.method_stats[method]["total_time"] += processing_time
            self.method_stats[method]["avg_time"] = self.method_stats[method]["total_time"] / self.method_stats[method]["count"]
            
            if success:
                self.method_stats[method]["success_count"] += 1
                
            # 오류 통계
            if error_type and not success:
                if error_type not in self.error_stats:
                    self.error_stats[error_type] = {"count": 0, "avg_time": 0, "total_time": 0}
                self.error_stats[error_type]["count"] += 1
                self.error_stats[error_type]["total_time"] += processing_time
                self.error_stats[error_type]["avg_time"] = self.error_stats[error_type]["total_time"] / self.error_stats[error_type]["count"]
                
                # 로그에 오류 상세 기록
                self._log_error_details(error_type, domain, method, processing_time)
                
        except Exception as e:
            print(f"통계 기록 오류: {e}")
    
    def record_template_usage(self, template_type: str, usage_count: int = 1):
        """템플릿 사용 통계"""
        try:
            if template_type not in self.template_usage_stats:
                self.template_usage_stats[template_type] = {"count": 0, "success_rate": 0.0, "last_used": None}
            self.template_usage_stats[template_type]["count"] += usage_count
            self.template_usage_stats[template_type]["last_used"] = datetime.now().isoformat()
        except Exception as e:
            print(f"템플릿 사용 통계 기록 오류: {e}")
    
    def record_memory_snapshot(self):
        """메모리 상태 기록"""
        if not self.memory_monitor_enabled:
            return
            
        try:
            memory_info = psutil.virtual_memory()
            process_info = psutil.Process()
            
            snapshot = {
                "timestamp": time.time(),
                "system_used_percent": memory_info.percent,
                "system_available_mb": memory_info.available / 1024 / 1024,
                "process_memory_mb": process_info.memory_info().rss / 1024 / 1024,
                "process_cpu_percent": process_info.cpu_percent()
            }
            
            self.memory_snapshots.append(snapshot)
            
            # 메모리 사용량이 85% 이상이면 로그에 경고 기록
            if memory_info.percent > 85:
                self._log_memory_warning(snapshot)
                
        except Exception as e:
            print(f"메모리 스냅샷 기록 오류: {e}")
    
    def generate_final_statistics(self, learning_data: Dict) -> Dict:
        """최종 통계 생성"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            total_questions = len(self.processing_times)
            
            # 기본 통계
            stats = {
                "execution_summary": {
                    "total_time_seconds": round(total_time, 2),
                    "total_time_minutes": round(total_time / 60, 2),
                    "total_questions": total_questions,
                    "avg_processing_time": round(sum(self.processing_times) / len(self.processing_times), 3) if self.processing_times else 0,
                    "questions_per_minute": round((total_questions / (total_time / 60)), 2) if total_time > 0 else 0,
                    "session_started": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None
                },
                "domain_analysis": self._analyze_domains(),
                "method_analysis": self._analyze_methods(),
                "performance_metrics": self._calculate_performance_metrics(),
                "learning_metrics": self._analyze_learning_data(learning_data),
                "system_metrics": self._analyze_system_metrics(),
                "error_analysis": self._analyze_errors(),
                "template_analysis": self.template_usage_stats if self.template_usage_stats else {},
                "quality_metrics": self._calculate_quality_metrics()
            }
            
            self._write_detailed_log(stats)
            return stats
            
        except Exception as e:
            print(f"최종 통계 생성 오류: {e}")
            return self._generate_fallback_stats()
    
    def _clear_log_and_write_system_info(self):
        """로그 초기화 및 시스템 정보 기록"""
        try:
            # 기존 로그 파일 내용 제거 (덮어쓰기 모드)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 새로운 세션 시작 ===\n")
                
                if self.memory_monitor_enabled:
                    f.write(f"[{timestamp}] 시스템 메모리: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB\n")
                    f.write(f"[{timestamp}] CPU 코어: {psutil.cpu_count()}개\n")
                    f.write(f"[{timestamp}] 메모리 모니터링: 활성화\n")
                else:
                    f.write(f"[{timestamp}] 메모리 모니터링: 비활성화 (psutil 없음)\n")
                    
                f.write(f"[{timestamp}] Python 버전: {sys.version.split()[0]}\n")
                f.flush()
                
        except Exception as e:
            print(f"로그 초기화 및 시스템 정보 기록 오류: {e}")
    
    def _log_error_details(self, error_type: str, domain: str, method: str, processing_time: float):
        """오류 상세 정보 로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 오류 발생 ===\n")
                f.write(f"[{timestamp}] 오류 유형: {error_type}\n")
                f.write(f"[{timestamp}] 도메인: {domain}\n")
                f.write(f"[{timestamp}] 처리 방법: {method}\n")
                f.write(f"[{timestamp}] 처리 시간: {processing_time:.3f}초\n")
                f.flush()
        except Exception as e:
            print(f"오류 상세 로그 기록 실패: {e}")
    
    def _log_memory_warning(self, snapshot: Dict):
        """메모리 경고 로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 메모리 경고 ===\n")
                f.write(f"[{timestamp}] 시스템 메모리 사용률: {snapshot['system_used_percent']:.1f}%\n")
                f.write(f"[{timestamp}] 프로세스 메모리 사용량: {snapshot['process_memory_mb']:.1f}MB\n")
                f.write(f"[{timestamp}] CPU 사용률: {snapshot['process_cpu_percent']:.1f}%\n")
                f.flush()
        except Exception as e:
            print(f"메모리 경고 로그 기록 실패: {e}")
    
    def _analyze_domains(self) -> Dict:
        """도메인 분석"""
        if not self.domain_stats:
            return {}
            
        try:
            total_questions = sum(domain["count"] for domain in self.domain_stats.values())
            
            domain_analysis = {}
            for domain, stats in self.domain_stats.items():
                success_rate = (stats["success_count"] / stats["count"]) * 100 if stats["count"] > 0 else 0
                
                domain_analysis[domain] = {
                    "question_count": stats["count"],
                    "percentage": round((stats["count"] / total_questions) * 100, 1) if total_questions > 0 else 0,
                    "avg_processing_time": round(stats["avg_time"], 3),
                    "total_processing_time": round(stats["total_time"], 3),
                    "success_rate": round(success_rate, 1),
                    "success_count": stats["success_count"]
                }
            
            return domain_analysis
        except Exception as e:
            print(f"도메인 분석 오류: {e}")
            return {}
    
    def _analyze_methods(self) -> Dict:
        """처리 방법 분석"""
        if not self.method_stats:
            return {}
            
        try:
            method_analysis = {}
            for method, stats in self.method_stats.items():
                success_rate = (stats["success_count"] / stats["count"]) * 100 if stats["count"] > 0 else 0
                method_analysis[method] = {
                    "question_count": stats["count"],
                    "success_count": stats["success_count"],
                    "success_rate": round(success_rate, 1),
                    "avg_processing_time": round(stats["avg_time"], 3),
                    "total_processing_time": round(stats["total_time"], 3),
                    "reliability_score": round(success_rate * (stats["count"] / 100), 1)
                }
            
            return method_analysis
        except Exception as e:
            print(f"처리 방법 분석 오류: {e}")
            return {}
    
    def _analyze_errors(self) -> Dict:
        """오류 분석"""
        if not self.error_stats:
            return {}
            
        try:
            error_analysis = {}
            total_errors = sum(error["count"] for error in self.error_stats.values())
            
            for error_type, stats in self.error_stats.items():
                error_analysis[error_type] = {
                    "count": stats["count"],
                    "percentage": round((stats["count"] / total_errors) * 100, 1) if total_errors > 0 else 0,
                    "avg_processing_time": round(stats["avg_time"], 3),
                    "total_processing_time": round(stats["total_time"], 3)
                }
            
            return error_analysis
        except Exception as e:
            print(f"오류 분석 실패: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict:
        """성능 지표 계산"""
        if not self.processing_times:
            return {}
            
        try:
            processing_times = sorted(self.processing_times)
            
            return {
                "min_processing_time": round(min(processing_times), 3),
                "max_processing_time": round(max(processing_times), 3),
                "median_processing_time": round(processing_times[len(processing_times) // 2], 3),
                "p95_processing_time": round(processing_times[int(len(processing_times) * 0.95)], 3),
                "p99_processing_time": round(processing_times[int(len(processing_times) * 0.99)], 3),
                "std_deviation": round(self._calculate_std_deviation(processing_times), 3),
                "efficiency_score": self._calculate_efficiency_score(),
                "throughput_score": self._calculate_throughput_score(),
                "consistency_score": self._calculate_consistency_score()
            }
        except Exception as e:
            print(f"성능 지표 계산 오류: {e}")
            return {}
    
    def _calculate_quality_metrics(self) -> Dict:
        """품질 지표 계산"""
        try:
            total_questions = len(self.processing_times)
            total_success = sum(stats["success_count"] for stats in self.method_stats.values())
            total_errors = sum(stats["count"] for stats in self.error_stats.values())
            
            overall_success_rate = (total_success / total_questions) * 100 if total_questions > 0 else 0
            error_rate = (total_errors / total_questions) * 100 if total_questions > 0 else 0
            
            return {
                "overall_success_rate": round(overall_success_rate, 1),
                "error_rate": round(error_rate, 1),
                "template_utilization": len(self.template_usage_stats),
                "domain_coverage": len(self.domain_stats),
                "method_diversity": len(self.method_stats)
            }
        except Exception as e:
            print(f"품질 지표 계산 오류: {e}")
            return {}
    
    def _analyze_learning_data(self, learning_data: Dict) -> Dict:
        """학습 데이터 분석"""
        try:
            total_successful = learning_data.get("successful_answers", 0)
            total_failed = learning_data.get("failed_answers", 0)
            total_patterns = learning_data.get("question_patterns", 0)
            
            learning_rate = (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
            
            return {
                "successful_answers": total_successful,
                "failed_answers": total_failed,
                "pattern_records": total_patterns,
                "learning_rate": round(learning_rate, 1),
                "knowledge_accumulation": total_successful + total_patterns,
                "error_recovery_data": total_failed,
                "learning_efficiency": round((total_successful / len(self.processing_times)) * 100, 1) if self.processing_times else 0
            }
        except Exception as e:
            print(f"학습 데이터 분석 오류: {e}")
            return {}
    
    def _analyze_system_metrics(self) -> Dict:
        """시스템 지표 분석"""
        if not self.memory_snapshots or not self.memory_monitor_enabled:
            return {"monitoring_status": "disabled" if not self.memory_monitor_enabled else "no_data"}
            
        try:
            system_memory_usages = [snap["system_used_percent"] for snap in self.memory_snapshots]
            process_memory_usages = [snap["process_memory_mb"] for snap in self.memory_snapshots]
            cpu_usages = [snap.get("process_cpu_percent", 0) for snap in self.memory_snapshots]
            
            return {
                "monitoring_status": "active",
                "avg_system_memory_usage": round(sum(system_memory_usages) / len(system_memory_usages), 1),
                "peak_system_memory_usage": round(max(system_memory_usages), 1),
                "avg_process_memory_usage": round(sum(process_memory_usages) / len(process_memory_usages), 1),
                "peak_process_memory_usage": round(max(process_memory_usages), 1),
                "avg_cpu_usage": round(sum(cpu_usages) / len(cpu_usages), 1),
                "peak_cpu_usage": round(max(cpu_usages), 1),
                "memory_snapshots": len(self.memory_snapshots),
                "system_stability": "정상" if max(system_memory_usages) < 85 else "주의필요"
            }
        except Exception as e:
            print(f"시스템 지표 분석 오류: {e}")
            return {"monitoring_status": "error"}
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0
            
        try:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
        except Exception:
            return 0
    
    def _calculate_efficiency_score(self) -> float:
        """처리 효율 점수 계산"""
        if not self.processing_times:
            return 0
            
        try:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            consistency = 1 - (self._calculate_std_deviation(self.processing_times) / avg_time) if avg_time > 0 else 0
            speed_score = max(0, 1 - (avg_time / 10))  # 10초를 기준으로 속도 점수
            
            return round((consistency * 0.6 + speed_score * 0.4) * 100, 1)
        except Exception:
            return 0
    
    def _calculate_throughput_score(self) -> float:
        """처리량 점수 계산"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            total_questions = len(self.processing_times)
            
            if total_time > 0:
                questions_per_minute = (total_questions / (total_time / 60))
                # 분당 30문항을 100점으로 기준
                return round(min((questions_per_minute / 30) * 100, 100), 1)
            return 0
        except Exception:
            return 0
    
    def _calculate_consistency_score(self) -> float:
        """일관성 점수 계산"""
        if not self.processing_times or len(self.processing_times) < 2:
            return 0
            
        try:
            std_dev = self._calculate_std_deviation(self.processing_times)
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            if avg_time > 0:
                consistency_ratio = 1 - (std_dev / avg_time)
                return round(max(0, consistency_ratio) * 100, 1)
            return 0
        except Exception:
            return 0
    
    def _generate_fallback_stats(self) -> Dict:
        """대체 통계 생성"""
        return {
            "execution_summary": {
                "total_time_seconds": 0,
                "total_time_minutes": 0,
                "total_questions": 0,
                "avg_processing_time": 0,
                "questions_per_minute": 0
            },
            "error": "통계 생성 중 오류 발생"
        }
    
    def _write_detailed_log(self, stats: Dict):
        """상세 로그 작성"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 실행 요약
                exec_summary = stats.get("execution_summary", {})
                f.write(f"[{timestamp}] === 실행 통계 ===\n")
                f.write(f"[{timestamp}] 총 처리 시간: {exec_summary.get('total_time_minutes', 0)}분 ({exec_summary.get('total_time_seconds', 0)}초)\n")
                f.write(f"[{timestamp}] 처리 문항 수: {exec_summary.get('total_questions', 0)}개\n")
                f.write(f"[{timestamp}] 평균 문항 처리 시간: {exec_summary.get('avg_processing_time', 0)}초\n")
                f.write(f"[{timestamp}] 분당 처리량: {exec_summary.get('questions_per_minute', 0)}문항/분\n")
                
                # 도메인 분석
                domain_analysis = stats.get("domain_analysis", {})
                if domain_analysis:
                    f.write(f"[{timestamp}] === 도메인별 분석 ===\n")
                    for domain, analysis in domain_analysis.items():
                        f.write(f"[{timestamp}] {domain}: {analysis.get('question_count', 0)}문항 ({analysis.get('percentage', 0)}%), "
                               f"평균 {analysis.get('avg_processing_time', 0)}초, 성공률 {analysis.get('success_rate', 0)}%\n")
                
                # 처리 방법 분석
                method_analysis = stats.get("method_analysis", {})
                if method_analysis:
                    f.write(f"[{timestamp}] === 처리 방법별 분석 ===\n")
                    for method, analysis in method_analysis.items():
                        f.write(f"[{timestamp}] {method}: {analysis.get('question_count', 0)}문항, "
                               f"성공률 {analysis.get('success_rate', 0)}%, 평균 {analysis.get('avg_processing_time', 0)}초, "
                               f"신뢰도 {analysis.get('reliability_score', 0)}점\n")
                
                # 성능 지표
                perf = stats.get("performance_metrics", {})
                if perf:
                    f.write(f"[{timestamp}] === 성능 지표 ===\n")
                    f.write(f"[{timestamp}] 처리 시간 범위: {perf.get('min_processing_time', 0)}초 ~ {perf.get('max_processing_time', 0)}초\n")
                    f.write(f"[{timestamp}] 중간값: {perf.get('median_processing_time', 0)}초, P95: {perf.get('p95_processing_time', 0)}초\n")
                    f.write(f"[{timestamp}] 표준편차: {perf.get('std_deviation', 0)}초, 효율점수: {perf.get('efficiency_score', 0)}점\n")
                    f.write(f"[{timestamp}] 처리량 점수: {perf.get('throughput_score', 0)}점, 일관성 점수: {perf.get('consistency_score', 0)}점\n")
                
                # 품질 지표
                quality = stats.get("quality_metrics", {})
                if quality:
                    f.write(f"[{timestamp}] === 품질 지표 ===\n")
                    f.write(f"[{timestamp}] 전체 성공률: {quality.get('overall_success_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 오류율: {quality.get('error_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 템플릿 활용: {quality.get('template_utilization', 0)}개\n")
                    f.write(f"[{timestamp}] 도메인 커버리지: {quality.get('domain_coverage', 0)}개\n")
                
                # 학습 지표
                learning = stats.get("learning_metrics", {})
                if learning:
                    f.write(f"[{timestamp}] === 학습 데이터 분석 ===\n")
                    f.write(f"[{timestamp}] 성공 답변: {learning.get('successful_answers', 0)}개\n")
                    f.write(f"[{timestamp}] 실패 답변: {learning.get('failed_answers', 0)}개\n")
                    f.write(f"[{timestamp}] 패턴 기록: {learning.get('pattern_records', 0)}개\n")
                    f.write(f"[{timestamp}] 학습 성공률: {learning.get('learning_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 학습 효율성: {learning.get('learning_efficiency', 0)}%\n")
                    f.write(f"[{timestamp}] 누적 지식량: {learning.get('knowledge_accumulation', 0)}개\n")
                
                # 시스템 지표
                system = stats.get("system_metrics", {})
                if system and system.get("monitoring_status") == "active":
                    f.write(f"[{timestamp}] === 시스템 지표 ===\n")
                    f.write(f"[{timestamp}] 평균 시스템 메모리 사용률: {system.get('avg_system_memory_usage', 0)}%\n")
                    f.write(f"[{timestamp}] 최대 시스템 메모리 사용률: {system.get('peak_system_memory_usage', 0)}%\n")
                    f.write(f"[{timestamp}] 평균 프로세스 메모리 사용량: {system.get('avg_process_memory_usage', 0)}MB\n")
                    f.write(f"[{timestamp}] 최대 프로세스 메모리 사용량: {system.get('peak_process_memory_usage', 0)}MB\n")
                    f.write(f"[{timestamp}] 평균 CPU 사용률: {system.get('avg_cpu_usage', 0)}%\n")
                    f.write(f"[{timestamp}] 시스템 상태: {system.get('system_stability', '알 수 없음')}\n")
                elif system:
                    f.write(f"[{timestamp}] === 시스템 지표 ===\n")
                    f.write(f"[{timestamp}] 모니터링 상태: {system.get('monitoring_status', '알 수 없음')}\n")
                
                # 오류 분석
                error_analysis = stats.get("error_analysis", {})
                if error_analysis:
                    f.write(f"[{timestamp}] === 오류 분석 ===\n")
                    for error_type, error_info in error_analysis.items():
                        f.write(f"[{timestamp}] {error_type}: {error_info.get('count', 0)}회 "
                               f"({error_info.get('percentage', 0)}%)\n")
                
                # 템플릿 사용 분석
                template_analysis = stats.get("template_analysis", {})
                if template_analysis:
                    f.write(f"[{timestamp}] === 템플릿 사용 분석 ===\n")
                    for template_type, template_info in template_analysis.items():
                        count = template_info.get('count', 0) if isinstance(template_info, dict) else template_info
                        f.write(f"[{timestamp}] {template_type}: {count}회\n")
                
                f.write(f"[{timestamp}] === 세션 종료 ===\n\n")
                f.flush()
                
        except Exception as e:
            print(f"상세 로그 작성 오류: {e}")
            # 기본 로그라도 작성 시도
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] 상세 로그 작성 실패: {e}\n")
                    f.write(f"[{timestamp}] === 세션 종료 (오류) ===\n\n")
            except Exception:
                pass