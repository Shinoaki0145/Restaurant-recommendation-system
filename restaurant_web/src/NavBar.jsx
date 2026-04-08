export default function NavBar() {
    return (
        <nav className="bg-surface dark:bg-zinc-900 sticky top-0 z-50">
            <div className="flex justify-between items-center w-full px-8 py-4 max-w-7xl mx-auto">
                <div className="flex items-center gap-8">
                    <span className="text-2xl font-bold tracking-tight text-orange-700 dark:text-orange-500 font-headline">Culinary Curator</span>
                    <div className="hidden md:flex items-center gap-6">
                        <a className="text-orange-700 dark:text-orange-500 font-bold border-b-2 border-orange-700 pb-1 text-sm tracking-wide" href="#">Explore</a>
                        <a className="text-zinc-600 dark:text-zinc-400 font-medium hover:text-orange-600 transition-colors duration-200 text-sm tracking-wide" href="#">Saved</a>
                        <a className="text-zinc-600 dark:text-zinc-400 font-medium hover:text-orange-600 transition-colors duration-200 text-sm tracking-wide" href="#">Profile</a>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <button className="p-2 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors duration-200">
                        <span className="material-symbols-outlined text-zinc-600 dark:text-zinc-400">notifications</span>
                    </button>
                    <button className="p-2 rounded-full hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors duration-200">
                        <span className="material-symbols-outlined text-zinc-600 dark:text-zinc-400">account_circle</span>
                    </button>
                </div>
            </div>
        </nav>

    )
}