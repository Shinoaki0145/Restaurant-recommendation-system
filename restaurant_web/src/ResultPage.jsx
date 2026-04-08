import ResultItem from "./ResultItem";

const sampleResults = [
  {
    id: "r1",
    name: "Osteria del Pane",
    image: "https://images.unsplash.com/photo-1528605248644-14dd04022da1",
    match_score: "96%",
    rating: 4.9,
    food_type: "Italian",
    distance: 1.2,
    description:
      "Handmade pasta, wood‑fired flavors, and a candlelit interior perfect for date night.",
  },
  {
    id: "r2",
    name: "Kinjo Modern Asian",
    image: "https://images.unsplash.com/photo-1414235077428-338989a2e8c0",
    match_score: "92%",
    rating: 4.7,
    food_type: "Modern Asian",
    distance: 2.4,
    description:
      "Bold seasonal tasting menus with a sleek, intimate setting.",
  },
  {
    id: "r3",
    name: "Neon Night Market",
    image: "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
    match_score: "89%",
    rating: 4.6,
    food_type: "Street Food",
    distance: 3.1,
    description:
      "Lively, casual spot serving authentic street flavors and late‑night bites.",
  },
  {
    id: "r4",
    name: "The Velvet Cut",
    image: "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
    match_score: "87%",
    rating: 4.8,
    food_type: "Steakhouse",
    distance: 1.5,
    description:
      "Premium cuts, vintage wines, and an upscale, classic atmosphere.",
  },
  {
    id: "r5",
    name: "Ocean's Edge",
    image: "https://images.unsplash.com/photo-1555396273-367ea4eb4db5",
    match_score: "85%",
    rating: 4.5,
    food_type: "Seafood",
    distance: 4.2,
    description:
      "Fresh daily catches served by the waterfront with panoramic ocean views.",
  },
  {
    id: "r6",
    name: "Verdant Kitchen",
    image: "https://images.unsplash.com/photo-1552566626-52f8b828add9",
    match_score: "83%",
    rating: 4.4,
    food_type: "Vegan",
    distance: 0.8,
    description:
      "Innovative plant-based dishes emphasizing sustainability and vibrant flavors.",
  },
  {
    id: "r7",
    name: "Midnight & Rye",
    image: "https://images.unsplash.com/photo-1514933651103-005eec06c04b",
    match_score: "81%",
    rating: 4.3,
    food_type: "Cocktail Bar",
    distance: 1.0,
    description:
      "Craft cocktails paired with artisanal charcuterie in a moody speakeasy setting.",
  },
  {
    id: "r8",
    name: "Kiku Omakase",
    image: "https://images.unsplash.com/photo-1528605248644-14dd04022da1",
    match_score: "78%",
    rating: 4.6,
    food_type: "Sushi",
    distance: 2.7,
    description:
      "Traditional omakase experience focused on freshness, precision, and minimalism.",
  }
];

export default function ResultPage() {
  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      {/* Header Section */}
      <header className="mb-16">
        <h1 className="text-display-lg text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-on-surface">
          Curated Just for You
        </h1>
        <p className="text-body-lg text-on-surface-variant max-w-2xl leading-relaxed">
          Our AI analyzed {sampleResults.length} spots with your request
        </p>
      </header>
      {/* Results Grid */}
      <div className="flex flex-col gap-16">
        {sampleResults.map((item, index) => (<ResultItem key={item.id} item={item} index={index} locateRight={index % 2 === 0}/>))}

      </div>
    </div>

  );
}