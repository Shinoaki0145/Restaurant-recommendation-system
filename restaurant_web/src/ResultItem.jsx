import { useEffect, useRef, useState } from "react";
import './Result.css'

export default function ResultItem({ item, index, locateRight }) {
    const ref = useRef();
    const [visible, setVisible] = useState(false);

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
                className={`group grid md:grid-cols-12 gap-10 items-center fade-item ${visible ? "show" : ""}`}

                style={{ transitionDelay: `${index * 0.1}s` }}>

                <div className="md:col-span-7 relative">
                    <div className="aspect-[16/9] overflow-hidden rounded-xl bg-surface-container">
                        <img className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"  src={item.image} />
                    </div>
                    <div className="absolute top-6 right-6 glass-chip px-4 py-2 rounded-full border border-white/20 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-tertiary"></span>
                        <span className="text-sm font-bold text-on-tertiary-container tracking-tight">{item.match_score} Match</span>
                    </div>
                </div>
                <div className="md:col-span-5 md:pl-4">
                    <h2 className="text-3xl font-bold mb-3 tracking-tight">{item.name}</h2>
                    <div className="flex items-center gap-4 text-on-surface-variant mb-6 font-medium">
                        <div className="flex text-secondary items-center">
                            <span className="material-symbols-outlined" style={{ fontVariationSettings: '"FILL" 1', fontSize: 16 }}>star</span>
                            <span className="text-sm font-bold ml-1">{item.rating}</span>
                        </div>
                        <span className="w-1 h-1 rounded-full bg-outline-variant"/>
                        <span>{item.food_type}</span>
                        <span className="w-1 h-1 rounded-full bg-outline-variant"></span>
                        <span>{item.distance} miles away</span>
                    </div>
                    <div className="bg-surface-container-low p-6 rounded-xl mb-8 border-l-4 border-tertiary">
                        <p className="text-on-surface italic leading-relaxed text-sm">
                            {item.description}
                        </p>
                    </div>
                    
                    <div className="flex gap-4">
                        <button className="ai-shimmer text-white px-8 py-3 rounded-full font-bold text-sm flex-1 hover:opacity-90 transition-all active:scale-95">Book a Table</button>
                        <button className="bg-surface-container-highest text-on-surface px-8 py-3 rounded-full font-bold text-sm flex-1 hover:bg-surface-dim transition-all">View Details</button>
                    </div>
                </div>
            </div>
        )
    } else {
        return (
            <div ref={ref}
                className={`group grid md:grid-cols-12 gap-10 items-center fade-item ${visible ? "show" : ""}`}

                style={{ transitionDelay: `${index * 0.1}s` }}>

                <div className="md:col-span-7 md:order-2 relative">
                    <div className="aspect-[16/9] overflow-hidden rounded-xl bg-surface-container">
                        <img className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" src={item.image} />
                    </div>
                    <div className="absolute top-6 left-6 glass-chip px-4 py-2 rounded-full border border-white/20 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-tertiary" />
                        <span className="text-sm font-bold text-on-tertiary-container tracking-tight">{item.match_score} Match</span>
                    </div>
                </div>
                <div className="md:col-span-5 md:pr-4">
                    <h2 className="text-3xl font-bold mb-3 tracking-tight text-right">{item.name}</h2>
                    <div className="flex justify-end items-center gap-4 text-on-surface-variant mb-6 font-medium">
                        <span>{item.distance} miles away</span>
                        <span className="w-1 h-1 rounded-full bg-outline-variant" />
                        <span>{item.food_type}</span>
                        <span className="w-1 h-1 rounded-full bg-outline-variant" />
                        <div className="flex text-secondary items-center">
                            <span className="material-symbols-outlined" style={{ fontVariationSettings: '"FILL" 1', fontSize: 16 }}>star</span>
                            <span className="text-sm font-bold ml-1">{item.rating}</span>
                        </div>
                    </div>
                    <div className="bg-surface-container-low p-6 rounded-xl mb-8 border-r-4 border-tertiary">
                        <p className="text-on-surface italic leading-relaxed text-sm">
                            {item.description}
                        </p>
                    </div>

                    <div className="flex gap-4">
                        <button className="ai-shimmer text-white px-8 py-3 rounded-full font-bold text-sm flex-1 hover:opacity-90 transition-all active:scale-95">Book a Table</button>
                        <button className="bg-surface-container-highest text-on-surface px-8 py-3 rounded-full font-bold text-sm flex-1 hover:bg-surface-dim transition-all">View Details</button>
                    </div>
                </div>
            </div>

        )
    }


}