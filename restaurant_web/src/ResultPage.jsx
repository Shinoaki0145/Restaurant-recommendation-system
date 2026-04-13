import NavBar from "./NavBar";
import ResultItem from "./ResultItem";


export default function ResultPage({ results = [], isLoading = false }) {
  return (
    <>
    <NavBar></NavBar>
    <div className="max-w-7xl mx-auto px-8 py-12">
      {/* Header Section */}
      <header className="mb-3">
        <h1 className="text-display-lg text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-on-surface">
          Curated Just for You
        </h1>
        <p className="text-body-lg text-on-surface-variant max-w-2xl leading-relaxed">
          {isLoading
            ? "Our AI is searching for the best spots..."
            : `Our AI analyzed ${results.length} spots with your request`}
        </p>
      </header>
      {/* Results Grid */}
      <div className="flex flex-col gap-16">
        {isLoading && (
          <div className="bg-surface-container-low rounded-xl p-8 text-on-surface-variant text-center">
            Loading recommendations...
          </div>
        )}

        {!isLoading && results.length === 0 && (
          <div className="bg-surface-container-low rounded-xl p-8 text-on-surface-variant text-center">
            No recommendations found.
          </div>
        )}

        {!isLoading && results.map((item, index) => (
          <ResultItem
            key={item.restaurant_id || item.id || index}
            item={item}
            index={index}
            locateRight={index % 2 === 0}
          />
        ))}

      </div>
    </div>
    </>

  );
}