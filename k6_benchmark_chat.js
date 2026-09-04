import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Counter } from "k6/metrics";

// ==== CẤU HÌNH ====
const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const VUS = parseInt(__ENV.VUS || "25");
const DURATION = __ENV.DURATION || "3m30s";

export const options = {
  vus: VUS,
  duration: DURATION,
  thresholds: {
    http_req_duration: ["p(95)<120000"],
  },
};

// ĐÃ SỬA: mỗi câu nêu rõ đúng 1 tên sản phẩm đầy đủ (khớp exact trong
// catalog), dạng hỏi thông số/tính năng đơn giản (FACT) -- KHÔNG dùng câu
// hỏi không nêu tên sản phẩm hay câu hỏi sự cố/FAQ chung chung, vì 2 loại
// đó có thể kích hoạt thêm lệnh gọi LLM phụ (cơ chế hỏi lại khi mơ hồ,
// hoặc tổng hợp nhiều sản phẩm) -- không so sánh công bằng được với số
// liệu benchmark cũ (đo trước khi có các cơ chế này). Đã phủ đủ cả 10
// sản phẩm khác nhau, mỗi sản phẩm 1 câu.
const QUESTIONS = [
  "Nguồn cấp của Bộ điều khiển hồng ngoại thông minh V2 là gì?",
  "Hướng dẫn cách đưa cảm biến cửa Mesh về chế độ sẵn sàng kết nối?",
  "Công tắc thông minh có tiếp điểm relay bao nhiêu A?",
  "Công tắc thông minh dimmer hỗ trợ công suất tối đa bao nhiêu W?",
  "Cảm biến hiện diện giám sát tối đa bao nhiêu người cùng lúc?",
  "Công tắc thông minh cho cửa cuốn, cửa cổng có bao nhiêu kênh đầu ra?",
  "Động Cơ Rèm hỗ trợ tải tối đa bao nhiêu kg?",
  "Bộ giám sát tiêu thụ điện năng thông minh hỗ trợ công suất tối đa bao nhiêu?",
  "Bộ điều khiển trung tâm - Gateway có bao nhiêu đèn báo trạng thái?",
  "Công tắc chống giật cho BNN có tiếp điểm relay bao nhiêu A?",
];

const chatLatency = new Trend("chat_latency_ms", true);
const requestsCompleted = new Counter("requests_completed");
const requestsFailed = new Counter("requests_failed");

export default function () {
  const question = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const payload = JSON.stringify({ question, debug: false });
  const params = {
    headers: { "Content-Type": "application/json" },
    timeout: "120s",
  };

  const start = Date.now();
  const res = http.post(`${BASE_URL}/api/chat`, payload, params);
  const elapsed = Date.now() - start;

  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "co answer": (r) => {
      try {
        return JSON.parse(r.body).answer.length > 0;
      } catch (e) {
        return false;
      }
    },
  });

  chatLatency.add(elapsed);
  if (ok) {
    requestsCompleted.add(1);
  } else {
    requestsFailed.add(1);
  }

  sleep(Math.random() * 2 + 1);
}

export function handleSummary(data) {
  const m = data.metrics;
  const latency = m.chat_latency_ms;
  const completed = m.requests_completed ? m.requests_completed.values.count : 0;
  const failed = m.requests_failed ? m.requests_failed.values.count : 0;

  const summary = {
    thoi_gian_chay: DURATION,
    so_nguoi_dung_dong_thoi: VUS,
    do_tre_trung_vi_ms: latency ? Math.round(latency.values.med) : null,
    do_tre_p95_ms: latency ? Math.round(latency.values["p(95)"]) : null,
    do_tre_trung_binh_ms: latency ? Math.round(latency.values.avg) : null,
    so_yeu_cau_thanh_cong: completed,
    so_yeu_cau_that_bai: failed,
  };

  console.log("\n===== KẾT QUẢ BENCHMARK =====");
  console.log(JSON.stringify(summary, null, 2));
  console.log("==============================\n");

  return {
    "stdout": "",
    "benchmark_result.json": JSON.stringify(summary, null, 2),
  };
}
