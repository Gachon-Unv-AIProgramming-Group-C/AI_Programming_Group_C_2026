const test = async () => {
  const payload = {
    premise: "대만 화롄 지역에서 규모 7.2의 강진이 발생한 시점은 2024년 4월 2일(현지 시각 3일)입니다.",
    hypothesis: "네, 2024년 4월 2일(현지 시각 3일) 대만 화롄 지역에서 규모 7.2의 강진이 발생했습니다."
  };
  const res = await fetch("http://127.0.0.1:8001/nli", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  console.log(JSON.stringify(data, null, 2));
};
test();
