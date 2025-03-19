import React from 'react';
import { Link } from 'react-router-dom';
import { Building2, Heart, Settings } from 'lucide-react';

export function Dashboard() {
  const [user] = React.useState(null);
  const [savedProperties] = React.useState([]);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // supabase.auth.getUser().then(({ data: { user } }) => {
    //   setUser(user);
    //   if (user) {
    //     getSavedProperties(user.id)
    //       .then(setSavedProperties)
    //       .finally(() => setLoading(false));
    //   }
    setLoading(false); // Set loading to false directly for now
  }, []);

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Please sign in to access your dashboard</h1>
          <Link
            to="/login"
            className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
          >
            Sign In
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-center mb-6">
              <div className="w-24 h-24 rounded-full bg-gray-200 mx-auto mb-4"></div>
              <h2 className="text-xl font-semibold">{user.email}</h2>
            </div>
            
            <nav className="space-y-2">
              <Link
                to="/dashboard"
                className="flex items-center space-x-2 p-2 bg-blue-50 text-blue-600 rounded-lg"
              >
                <Building2 className="h-5 w-5" />
                <span>My Properties</span>
              </Link>
              <Link
                to="/dashboard/saved"
                className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded-lg"
              >
                <Heart className="h-5 w-5" />
                <span>Saved Properties</span>
              </Link>
              <Link
                to="/dashboard/settings"
                className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded-lg"
              >
                <Settings className="h-5 w-5" />
                <span>Settings</span>
              </Link>
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold mb-6">My Saved Properties</h2>
            
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {savedProperties.map((saved) => {
                  const property = saved.properties;
                  return (
                    <Link
                      key={property.id}
                      to={`/property/${property.id}`}
                      className="bg-white rounded-lg shadow-sm hover:shadow-md transition overflow-hidden"
                    >
                      <div className="aspect-w-16 aspect-h-9">
                        <img
                          src={property.property_images[0]?.url || 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6'}
                          alt={property.title}
                          className="object-cover w-full h-full"
                        />
                      </div>
                      <div className="p-4">
                        <h3 className="font-semibold text-lg mb-2">{property.title}</h3>
                        <p className="text-blue-600 font-semibold">
                          ${property.price.toLocaleString()}
                        </p>
                        <p className="text-gray-600 text-sm mt-2">
                          {property.city}, {property.state}
                        </p>
                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}