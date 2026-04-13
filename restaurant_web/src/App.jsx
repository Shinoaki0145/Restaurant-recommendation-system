import './App.css'
import { useState } from 'react';
import Home from './Home';
import { Routes, Route } from "react-router-dom";
import ResultPage from './ResultPage';

const sampleResults = [
  {
      "restaurant_id": "676469",
      "rank_score": 3.3443357944488525,
      "restaurant": {
        "restaurant_id": "676469",
        "restaurant_name_meta": "Apuro - Ẩm Thực Hàn Quốc",
        "address_meta": "8A/13C1 Thái Văn Lung, P. Bến Nghé",
        "district_meta": "Quận 1",
        "area_meta": "Parkson Lê Thánh Tôn",
        "meta_keywords": "quán ăn hàn quốc, quán ăn hàn quốc quận 1, quán ăn hàn quốc thái văn lung, quán ăn hàn quốc ngon quận 1, quán ăn hàn quốc ngon, ẩm thực hàn quốc quận 1, ẩm thực hàn quốc thái văn lung, ăn tối thái văn lung, quán rựu thái văn lung,",
        "cuisines_meta": "Món Hàn",
        "target_audience_raw": "Nhóm hội || Giới văn phòng || Khách du lịch",
        "category_raw": "Quán ăn",
        "restaurant_url": "ho-chi-minh/apuro-am-thuc-han-quoc",
        "picturemodel": "https://down-vn.img.susercontent.com/vn-11134259-7r98o-lwf9o86jnheh8d@resize_ss576x330",
        "pricemin": 20000,
        "pricemax": 500000,
        "vi_tri": 6.43,
        "gia_ca": 6.29,
        "chat_luong": 7.43,
        "phuc_vu": 6.57,
        "khong_gian": 6.71,
        "excellent": 0,
        "good": 4,
        "average": 1,
        "bad": 2,
        "totalview": 5188,
        "totalfavourite": 58,
        "totalcheckins": 3,
        "delivery_flag": 0,
        "booking_flag": 0,
        "rest_days_raw": "",
        "thu_hai": "('11:30', '01:00')",
        "thu_ba": "('11:30', '01:00')",
        "thu_tu": "('11:30', '01:00')",
        "thu_nam": "('11:30', '01:00')",
        "thu_sau": "('11:30', '01:00')",
        "thu_bay": "('11:30', '01:00')",
        "chu_nhat": "('11:30', '01:00')"
      },
      "retrieval_score": 0.835535944
    },
    {
      "restaurant_id": "21148",
      "rank_score": 3.2171261310577393,
      "restaurant": {
        "restaurant_id": "21148",
        "restaurant_name_meta": "Yeije - Ẩm Thực Hàn Quốc",
        "address_meta": "Tầng 3, Bitexco Tower, 2 Hải Triều",
        "district_meta": "Quận 1",
        "area_meta": "Bitexco Tower",
        "meta_keywords": "nhà hàng go cung, nhà hàng go cung đường hải triều, nhà hàng go cung, Go Cung - Traditional Korean Restaurant, nhà hàng hàn quốc, món ăn hàn quốc, món ngon, thịt nướng hàn quốc, nhà hàng hàn quốc yeije, quán ăn hàn quốc quận 1",
        "cuisines_meta": "Món Hàn",
        "target_audience_raw": "Cặp đôi || Gia đình || Nhóm hội || Giới văn phòng || Giới Manager",
        "category_raw": "Nhà hàng",
        "restaurant_url": "ho-chi-minh/go-cung-traditional-korean-restaurant",
        "picturemodel": "https://down-vn.img.susercontent.com/vn-11134259-7r98o-lwcjs9nx5k2j44@resize_ss576x330",
        "pricemin": 50000,
        "pricemax": 220000,
        "vi_tri": 7.62,
        "gia_ca": 6.92,
        "chat_luong": 8.19,
        "phuc_vu": 7.65,
        "khong_gian": 7.62,
        "excellent": 2,
        "good": 22,
        "average": 2,
        "bad": 0,
        "totalview": 47338,
        "totalfavourite": 39,
        "totalcheckins": 17,
        "delivery_flag": 0,
        "booking_flag": 1,
        "rest_days_raw": "",
        "thu_hai": "('09:30', '22:00')",
        "thu_ba": "('09:30', '22:00')",
        "thu_tu": "('09:30', '22:00')",
        "thu_nam": "('09:30', '22:00')",
        "thu_sau": "('09:30', '22:00')",
        "thu_bay": "('09:30', '22:00')",
        "chu_nhat": "('09:30', '22:00')"
      },
      "retrieval_score": 0.834692359
    },
    {
      "restaurant_id": "967474",
      "rank_score": 2.713068723678589,
      "restaurant": {
        "restaurant_id": "967474",
        "restaurant_name_meta": "Big - Korea Food - Món Ăn Hàn Quốc",
        "address_meta": "15 Phạm Viết Chánh",
        "district_meta": "Quận 1",
        "area_meta": "Từ Dũ",
        "meta_keywords": "quán ăn hàn quốc đường phạm viết chánh, quán ăn hàn quốc quận 1, nhà hàng hàn quốc đường phạm viết chánh, nhà hàng hàn quốc quận 1, món hàn đường phạm viết chánh, món hàn quận 1, big food",
        "cuisines_meta": "Món Hàn",
        "target_audience_raw": "Gia đình || Nhóm hội || Giới văn phòng",
        "category_raw": "Nhà hàng",
        "restaurant_url": "ho-chi-minh/big-korea-food-mon-an-han-quoc",
        "picturemodel": "https://down-vn.img.susercontent.com/vn-11134259-7r98o-lwfwyvfrz3dld5@resize_ss576x330",
        "pricemin": 50000,
        "pricemax": 170000,
        "vi_tri": 8.33,
        "gia_ca": 7.33,
        "chat_luong": 7.67,
        "phuc_vu": 7.67,
        "khong_gian": 6.33,
        "excellent": 2,
        "good": 0,
        "average": 0,
        "bad": 1,
        "totalview": 226,
        "totalfavourite": 6,
        "totalcheckins": 3,
        "delivery_flag": 1,
        "booking_flag": 0,
        "rest_days_raw": "",
        "thu_hai": "('09:45', '22:00')",
        "thu_ba": "('09:45', '22:00')",
        "thu_tu": "('09:45', '22:00')",
        "thu_nam": "('09:45', '22:00')",
        "thu_sau": "('09:45', '22:00')",
        "thu_bay": "('09:45', '22:00')",
        "chu_nhat": "('09:45', '22:00')"
      },
      "retrieval_score": 0.833725929
    },
    {
      "restaurant_id": "1049614",
      "rank_score": 2.4191622734069824,
      "restaurant": {
        "restaurant_id": "1049614",
        "restaurant_name_meta": "Seoul House - Quán Ăn Hàn Quốc 1",
        "address_meta": "46 - 48 Bùi Thị Xuân, P. Bến Thành",
        "district_meta": "Quận 1",
        "area_meta": "Galaxy Nguyễn Du",
        "meta_keywords": "seoul, seoul house, seoul house bùi thị xuân, quán ăn hàn quốc số 1, quán hàn bùi thị xuân, món hàn ngon bùi thị xuân, seoul quận 1, seoul house quận 1, seoul quán ăn hàn quốc quận 1, món hàn ngon quận 1, nhà hàng hàn quốc quận 1, 46 bùi thị xuân, 48 bùi thị xuân",
        "cuisines_meta": "Món Hàn",
        "target_audience_raw": "Sinh viên || Gia đình || Nhóm hội",
        "category_raw": "Nhà hàng",
        "restaurant_url": "ho-chi-minh/seoul-house-quan-an-han-quoc-1",
        "picturemodel": "https://down-vn.img.susercontent.com/vn-11134259-7r98o-lw9fh8ksm8yhc2@resize_ss576x330",
        "pricemin": 120000,
        "pricemax": 600000,
        "vi_tri": 10,
        "gia_ca": 9.5,
        "chat_luong": 9.5,
        "phuc_vu": 7.5,
        "khong_gian": 7.5,
        "excellent": 1,
        "good": 1,
        "average": 0,
        "bad": 0,
        "totalview": 1696,
        "totalfavourite": 1,
        "totalcheckins": 0,
        "delivery_flag": 1,
        "booking_flag": 0,
        "rest_days_raw": "",
        "thu_hai": "('10:00', '21:00')",
        "thu_ba": "('10:00', '21:00')",
        "thu_tu": "('10:00', '21:00')",
        "thu_nam": "('10:00', '21:00')",
        "thu_sau": "('10:00', '21:00')",
        "thu_bay": "('10:00', '21:00')",
        "chu_nhat": "('10:00', '21:00')"
      },
      "retrieval_score": 0.832120478
    },
    {
      "restaurant_id": "711015",
      "rank_score": 2.333174705505371,
      "restaurant": {
        "restaurant_id": "711015",
        "restaurant_name_meta": "Nhà Hàng Hàn Quốc Jin Sun Dae",
        "address_meta": "Lầu 1, 17 Thái Văn Lung, P. Bến Nghé",
        "district_meta": "Quận 1",
        "area_meta": "Bến Bạch Đằng",
        "meta_keywords": "jin sun dae, nhà hàng jin sun dae, nhà hàng hàn quốc jin sun dae, nhà hàng hàn quốc thái văn lung, jin sun dae thái văn lung, jin sun dae quận 1",
        "cuisines_meta": "Món Hàn",
        "target_audience_raw": "Sinh viên || Gia đình || Giới văn phòng",
        "category_raw": "Nhà hàng",
        "restaurant_url": "ho-chi-minh/nha-hang-han-quoc-jin-sun-dae",
        "picturemodel": "https://down-vn.img.susercontent.com/vn-11134259-7r98o-lw9jyb8103ix91@resize_ss576x330",
        "pricemin": 50000,
        "pricemax": 150000,
        "vi_tri": 8.25,
        "gia_ca": 8,
        "chat_luong": 9,
        "phuc_vu": 8.25,
        "khong_gian": 7.25,
        "excellent": 0,
        "good": 4,
        "average": 0,
        "bad": 0,
        "totalview": 2244,
        "totalfavourite": 34,
        "totalcheckins": 2,
        "delivery_flag": 0,
        "booking_flag": 0,
        "rest_days_raw": "",
        "thu_hai": "('09:00', '21:00')",
        "thu_ba": "('09:00', '21:00')",
        "thu_tu": "('09:00', '21:00')",
        "thu_nam": "('09:00', '21:00')",
        "thu_sau": "('09:00', '21:00')",
        "thu_bay": "('09:00', '21:00')",
        "chu_nhat": "('09:00', '21:00')"
      },
      "retrieval_score": 0.834715307
    }
];


function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchRecommendations = async (query) => {
    const response = await fetch("http://localhost:8000/rank", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        top_k: 5,
        pinecone_top_k: 30,
        use_pinecone: true,
        force_retrain: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();

    return Array.isArray(data?.results) ? data.results : [];
  };

  const handleSearch = async (query) => {
    setIsLoading(true);
    try {
      const apiResults = await fetchRecommendations(query);
      setResults(Array.isArray(apiResults) ? apiResults : []);
    } catch (error) {
      console.error("Failed to fetch recommendations:", error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Routes>
      <Route path="/" element={<Home onSearch={handleSearch} isLoading={isLoading} />} />
      <Route path="/result" element={<ResultPage results={results} isLoading={isLoading} />} />
    </Routes>
  );
}

export default App
