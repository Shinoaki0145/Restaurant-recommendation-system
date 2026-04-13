import { useEffect, useRef, useState } from "react";
import './Result.css'

export default function ResultItem({ item, index, locateRight }) {
    const ref = useRef();
    const [visible, setVisible] = useState(false);
    const restaurant = item?.restaurant ?? {};

    const image = restaurant.picturemodel || item?.image;
    const name = restaurant.restaurant_name_meta || item?.name || "Unnamed restaurant";
    const address = [restaurant.address_meta, restaurant.district_meta].filter(Boolean).join(", ");
    const targetAudience = (restaurant.target_audience_raw || "")
        .split("||")
        .map((value) => value.trim())
        .filter(Boolean)
        .join(" | ");
    const cuisine = restaurant.cuisines_meta || item?.food_type || "N/A";
    const hasDelivery = Number(restaurant.delivery_flag) === 1;
    const hasBooking = Number(restaurant.booking_flag) === 1;
    const metrics = {
        viTri: restaurant.vi_tri ?? "-",
        giaCa: restaurant.gia_ca ?? "-",
        chatLuong: restaurant.chat_luong ?? "-",
        phucVu: restaurant.phuc_vu ?? "-",
        khongGian: restaurant.khong_gian ?? "-",
    };
    const stats = {
        excellent: restaurant.excellent ?? 0,
        good: restaurant.good ?? 0,
        average: restaurant.average ?? 0,
        bad: restaurant.bad ?? 0,
        totalview: restaurant.totalview ?? 0,
        totalfavourite: restaurant.totalfavourite ?? 0,
        totalcheckins: restaurant.totalcheckins ?? 0,
    };

    const formatPrice = (value) => {
        if (typeof value !== "number") return null;
        return `${value.toLocaleString("vi-VN")}đ`;
    };

    const minPrice = formatPrice(restaurant.pricemin);
    const maxPrice = formatPrice(restaurant.pricemax);
    const priceRange = minPrice && maxPrice ? `${minPrice} - ${maxPrice}` : "Đang cập nhật";

    const matchLabel = (() => {
        if (typeof item?.match_score === "string" && item.match_score.trim()) {
            return item.match_score;
        }

        if (typeof item?.retrieval_score === "number") {
            return `${Math.round(item.retrieval_score * 100)}%`;
        }

        return "--%";
    })();

    const formatCount = (value) => {
        if (typeof value !== "number") return "0";
        return value.toLocaleString("vi-VN");
    };

    useEffect(() => {
        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                setVisible(true);
            } else {
                setVisible(false)
            }
        }, { threshold: 0 });


        observer.observe(ref.current);

        return () => observer.disconnect();
    }, []);


    if (locateRight) {
        return (
            <div ref={ref}
                className={`group grid md:grid-cols-12 gap-10 items-start fade-item ${visible ? "show" : ""}`}

                style={{ transitionDelay: `${index * 0.1}s` }}>

                <div className="md:col-span-6 md:self-center relative">
                    <div className="aspect-[4/3] overflow-hidden rounded-xl bg-surface-container">
                        <img
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                            src={image}
                            alt={name}
                        />
                    </div>
                    <div className="absolute top-6 right-6 glass-chip px-4 py-2 rounded-full border border-white/20 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full ai-shimmer"></span>
                        <span className="text-sm font-bold text-on-surface tracking-tight">{matchLabel} Match</span>
                    </div>
                </div>
                <div className="md:col-span-6 md:pl-4">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            {hasDelivery && (
                                <span className="bg-green-100 text-green-700 text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded flex items-center gap-1">
                                    <span className="material-symbols-outlined !text-[12px]">delivery_dining</span>
                                    Giao hàng tận nơi
                                </span>
                            )}
                            {hasBooking && (
                                <span className="bg-blue-100 text-blue-700 text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded flex items-center gap-1">
                                    <span className="material-symbols-outlined !text-[12px]">event_available</span>
                                    Đặt trước
                                </span>
                            )}
                        </div>
                    </div>
                    <h2 className="text-3xl font-bold mb-1 tracking-tight">{name}</h2>
                    <p className="text-on-surface-variant font-medium mb-3">{address || "Đang cập nhật"}</p>
                    {targetAudience && (
                        <p className="text-on-surface-variant/80 text-sm mb-4">{targetAudience}</p>
                    )}
                    <div className="flex flex-wrap items-center gap-4 text-on-surface-variant mb-6 text-sm">
                        <span className="font-semibold text-primary">{cuisine}</span>
                        <span className="w-1 h-1 rounded-full bg-outline-variant"></span>
                        <span className="font-medium">{priceRange}</span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-4 gap-x-6 mb-8 bg-surface-container-low p-6 rounded-xl">
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">location_on</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Vị trí</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.viTri}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">payments</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Giá cả</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.giaCa}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">restaurant</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Chất lượng</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.chatLuong}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">person</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Phục vụ</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.phucVu}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">grid_view</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Không gian</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.khongGian}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">visibility</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Lượt xem</span>
                            </div>
                            <span className="text-lg font-bold">{formatCount(stats.totalview)}</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-8">
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-emerald-700">Excellent</p>
                            <p className="text-base font-bold">{formatCount(stats.excellent)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-lime-700">Good</p>
                            <p className="text-base font-bold">{formatCount(stats.good)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-amber-700">Average</p>
                            <p className="text-base font-bold">{formatCount(stats.average)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-rose-700">Bad</p>
                            <p className="text-base font-bold">{formatCount(stats.bad)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-on-surface-variant">Favourites</p>
                            <p className="text-base font-bold">{formatCount(stats.totalfavourite)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-on-surface-variant">Check-ins</p>
                            <p className="text-base font-bold">{formatCount(stats.totalcheckins)}</p>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <a
                            className="ai-shimmer text-white px-8 py-3 rounded-full font-bold text-sm flex-1 inline-flex items-center justify-center text-center hover:opacity-90 transition-all active:scale-95 shadow-lg shadow-primary/20"
                            href={`https://www.foody.vn/${restaurant.restaurant_url}`}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            More information
                        </a>
                    </div>
                </div>
            </div>
        )
    } else {
        return (
            <div ref={ref}
                className={`group grid md:grid-cols-12 gap-10 items-start fade-item ${visible ? "show" : ""}`}

                style={{ transitionDelay: `${index * 0.1}s` }}>

                <div className="md:col-span-6 md:order-2 md:self-center relative">
                    <div className="aspect-[4/3] overflow-hidden rounded-xl bg-surface-container">
                        <img
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                            src={image}
                            alt={name}
                        />
                    </div>
                    <div className="absolute top-6 left-6 glass-chip px-4 py-2 rounded-full border border-white/20 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full ai-shimmer" />
                        <span className="text-sm font-bold text-on-surface tracking-tight">{matchLabel} Match</span>
                    </div>
                </div>
                <div className="md:col-span-6 md:pr-4">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            {hasDelivery && (
                                <span className="bg-green-100 text-green-700 text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded flex items-center gap-1">
                                    <span className="material-symbols-outlined !text-[12px]">delivery_dining</span>
                                    Giao hàng tận nơi
                                </span>
                            )}
                            {hasBooking && (
                                <span className="bg-blue-100 text-blue-700 text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded flex items-center gap-1">
                                    <span className="material-symbols-outlined !text-[12px]">event_available</span>
                                    Đặt trước
                                </span>
                            )}
                        </div>
                    </div>
                    <h2 className="text-3xl font-bold mb-1 tracking-tight">{name}</h2>
                    <p className="text-on-surface-variant font-medium mb-3">{address || "Đang cập nhật"}</p>
                    {targetAudience && (
                        <p className="text-on-surface-variant/80 text-sm mb-4">{targetAudience}</p>
                    )}
                    <div className="flex flex-wrap items-center gap-4 text-on-surface-variant mb-6 text-sm">
                        <span className="font-semibold text-primary">{cuisine}</span>
                        <span className="w-1 h-1 rounded-full bg-outline-variant" />
                        <span className="font-medium">{priceRange}</span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-4 gap-x-6 mb-8 bg-surface-container-low p-6 rounded-xl">
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">location_on</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Vị trí</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.viTri}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">payments</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Giá cả</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.giaCa}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">restaurant</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Chất lượng</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.chatLuong}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">person</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Phục vụ</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.phucVu}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">grid_view</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Không gian</span>
                            </div>
                            <span className="text-lg font-bold">{metrics.khongGian}</span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-1 text-on-surface-variant">
                                <span className="material-symbols-outlined !text-[18px]">visibility</span>
                                <span className="text-[11px] font-bold uppercase tracking-wider">Lượt xem</span>
                            </div>
                            <span className="text-lg font-bold">{formatCount(stats.totalview)}</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-8">
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-emerald-700">Excellent</p>
                            <p className="text-base font-bold">{formatCount(stats.excellent)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-lime-700">Good</p>
                            <p className="text-base font-bold">{formatCount(stats.good)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-amber-700">Average</p>
                            <p className="text-base font-bold">{formatCount(stats.average)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-rose-700">Bad</p>
                            <p className="text-base font-bold">{formatCount(stats.bad)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-on-surface-variant">Favourites</p>
                            <p className="text-base font-bold">{formatCount(stats.totalfavourite)}</p>
                        </div>
                        <div className="bg-surface-container-low rounded-lg px-3 py-2">
                            <p className="text-[11px] uppercase font-semibold tracking-wide text-on-surface-variant">Check-ins</p>
                            <p className="text-base font-bold">{formatCount(stats.totalcheckins)}</p>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <a
                            className="ai-shimmer text-white px-8 py-3 rounded-full font-bold text-sm flex-1 inline-flex items-center justify-center text-center hover:opacity-90 transition-all active:scale-95 shadow-lg shadow-primary/20"
                            href={`https://www.foody.vn/${restaurant.restaurant_url}`}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            More information
                        </a>
                    </div>
                </div>
            </div>

        )
    }


}